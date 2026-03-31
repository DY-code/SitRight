"""
PySide6 坐姿矫正辅助系统
迁移自 Tkinter 版本，性能优化，UI更流畅
"""

import os
import sys

from PySide6.QtCore import QLibraryInfo


def _pin_pyside6_qt_plugins():
    """强制 Qt 使用 PySide6 的插件目录，规避 OpenCV 的 Qt 插件冲突。"""
    plugins_root = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
    if not plugins_root:
        return
    os.environ["QT_PLUGIN_PATH"] = plugins_root
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(plugins_root, "platforms")


# 先固定 Qt 插件路径，再导入其他可能改写该路径的库（如 cv2）
_pin_pyside6_qt_plugins()

import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import threading
import time
import queue
import json
import random
import tempfile
import subprocess
import shutil
import hashlib
from datetime import date

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QCheckBox, QGroupBox, QFrame, QScrollArea, QComboBox, QProgressBar, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
from PySide6.QtGui import QImage, QPixmap, QPalette, QColor, QFont

# 某些 OpenCV 构建会在导入后覆盖 Qt 插件路径，这里再次锁定到 PySide6。
_pin_pyside6_qt_plugins()


_TTS_WORKER_STARTED = False
_TTS_WORKER_LOCK = threading.Lock()
_TTS_QUEUE = queue.Queue(maxsize=12)
_TTS_DEDUP_LOCK = threading.Lock()
_LAST_TTS_TEXT = ""
_LAST_TTS_TS = 0.0
_TTS_BACKEND = None
_TTS_DISABLED_NOTICE_PRINTED = False
_EDGE_GEN_TIMEOUT_SEC = 8
_EDGE_PROBE_TIMEOUT_SEC = 6
_EDGE_FAIL_LOCK = threading.Lock()
_EDGE_FAIL_COUNT = 0
_EDGE_DISABLED_UNTIL = 0.0
_EDGE_RETRY_BASE_SEC = 10
_EDGE_RETRY_MAX_SEC = 180
_EDGE_PRECACHE_STARTED = False
_EDGE_PRECACHE_LOCK = threading.Lock()
_TTS_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "tts_cache")
_EDGE_PRECACHE_TEXTS = [
    "坐姿风险较高，请立即调整。",
    "请注意坐姿。",
    "久坐提醒，请起身活动。",
    "摄像头连接异常，请检查摄像头是否可用。",
]


def _normalize_tts_text(text):
    if text is None:
        return ""
    normalized = str(text).strip()
    replace_map = {
        "⚠️": "",
        ":": "，",
        "：": "，",
        "/": "和",
        "、": "，",
        "!": "。",
        "！": "。",
        ";": "，",
        "；": "，",
    }
    for k, v in replace_map.items():
        normalized = normalized.replace(k, v)
    while "，，" in normalized:
        normalized = normalized.replace("，，", "，")
    if normalized and normalized[-1] not in "。！？":
        normalized += "。"
    return normalized


def _detect_tts_backend():
    """自动选择语音后端：仅使用 edge-tts。"""
    preferred = os.environ.get("POSTURE_TTS_BACKEND", "").strip().lower()
    if preferred == "pyttsx3":
        print("[语音播报] pyttsx3 已弃用，已忽略该设置。")
    if preferred == "edge":
        return preferred

    if shutil.which("edge-tts") and (shutil.which("gst-play-1.0") or shutil.which("paplay") or shutil.which("aplay")):
        if not _probe_edge_tts():
            print("[语音播报] edge-tts 连通性异常，先使用临时回退，稍后自动重试 edge。")
        return "edge"
    return "none"


def _probe_edge_tts():
    """快速探测 edge-tts 是否可用，避免首次播报长时间卡顿。"""
    edge_cmd = shutil.which("edge-tts")
    if not edge_cmd:
        return False
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmpf:
            tmp_path = tmpf.name
        cmd = [
            edge_cmd,
            "--voice", "zh-CN-XiaoxiaoNeural",
            "--text", "测试",
            "--write-media", tmp_path,
        ]
        proxy = os.environ.get("POSTURE_TTS_PROXY", "").strip()
        if proxy:
            cmd.extend(["--proxy", proxy])
        gen = subprocess.run(cmd, capture_output=True, text=True, timeout=_EDGE_PROBE_TIMEOUT_SEC)
        if gen.returncode != 0:
            return False
        return os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0
    except Exception:
        return False
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _edge_available_now():
    with _EDGE_FAIL_LOCK:
        return time.time() >= _EDGE_DISABLED_UNTIL


def _mark_edge_success():
    global _EDGE_FAIL_COUNT, _EDGE_DISABLED_UNTIL
    with _EDGE_FAIL_LOCK:
        _EDGE_FAIL_COUNT = 0
        _EDGE_DISABLED_UNTIL = 0.0


def _mark_edge_failure():
    global _EDGE_FAIL_COUNT, _EDGE_DISABLED_UNTIL
    with _EDGE_FAIL_LOCK:
        _EDGE_FAIL_COUNT += 1
        backoff = min(_EDGE_RETRY_MAX_SEC, _EDGE_RETRY_BASE_SEC * (2 ** (_EDGE_FAIL_COUNT - 1)))
        _EDGE_DISABLED_UNTIL = time.time() + backoff
        return _EDGE_FAIL_COUNT, int(backoff)


def _edge_cache_path(text):
    normalized = _normalize_tts_text(text)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return os.path.join(_TTS_CACHE_DIR, f"{digest}.mp3")


def _has_valid_audio_file(path):
    return bool(path) and os.path.exists(path) and os.path.getsize(path) > 0


def _build_edge_cmd(edge_cmd, text, output_path):
    cmd = [
        edge_cmd,
        "--voice", "zh-CN-XiaoxiaoNeural",
        "--rate=-6%",
        "--pitch=+2Hz",
        "--text", text,
        "--write-media", output_path,
    ]
    proxy = os.environ.get("POSTURE_TTS_PROXY", "").strip()
    if proxy:
        cmd.extend(["--proxy", proxy])
    return cmd


def _generate_edge_audio_file(edge_cmd, text, out_path, timeout_sec):
    try:
        os.makedirs(_TTS_CACHE_DIR, exist_ok=True)
        tmp_out = f"{out_path}.tmp"
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
        cmd = _build_edge_cmd(edge_cmd, text, tmp_out)
        gen = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        if gen.returncode != 0:
            err = (gen.stderr or "").strip()
            print(f"[语音播报] edge-tts 生成失败: {err}")
            return False
        if not _has_valid_audio_file(tmp_out):
            print("[语音播报] edge-tts 生成的音频文件为空。")
            return False
        os.replace(tmp_out, out_path)
        return True
    except Exception as e:
        print(f"[语音播报] edge-tts 播放异常: {e}")
        return False
    finally:
        tmp_out = f"{out_path}.tmp"
        if os.path.exists(tmp_out):
            try:
                os.remove(tmp_out)
            except Exception:
                pass


def _precache_edge_phrases_async():
    global _EDGE_PRECACHE_STARTED
    with _EDGE_PRECACHE_LOCK:
        if _EDGE_PRECACHE_STARTED:
            return
        _EDGE_PRECACHE_STARTED = True

    def run_precache():
        edge_cmd = shutil.which("edge-tts")
        if not edge_cmd:
            return
        for phrase in _EDGE_PRECACHE_TEXTS:
            text = _normalize_tts_text(phrase)
            cache_path = _edge_cache_path(text)
            if _has_valid_audio_file(cache_path):
                continue
            if not _edge_available_now():
                return
            ok = _generate_edge_audio_file(edge_cmd, text, cache_path, timeout_sec=max(_EDGE_GEN_TIMEOUT_SEC, 12))
            if ok:
                _mark_edge_success()
            else:
                fail_count, backoff = _mark_edge_failure()
                print(f"[语音播报] edge-tts 预缓存失败次数={fail_count}，{backoff}秒后重试。")
                return
        print("[语音播报] edge 常用语句缓存已就绪。")

    threading.Thread(target=run_precache, daemon=True).start()


def _play_audio_file(path):
    """按可用播放器顺序尝试播放音频文件。"""
    candidates = []
    if shutil.which("gst-play-1.0"):
        candidates.append(["gst-play-1.0", "-q", "--no-interactive", path])
    if shutil.which("paplay"):
        candidates.append(["paplay", path])
    if shutil.which("aplay"):
        candidates.append(["aplay", path])

    if not candidates:
        print("[语音播报] 未找到可用音频播放器（gst-play/paplay/aplay）。")
        return False

    for cmd in candidates:
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if res.returncode == 0:
                return True
            err = (res.stderr or "").strip()
            if err:
                print(f"[语音播报] 播放器失败 ({cmd[0]}): {err}")
        except Exception as e:
            print(f"[语音播报] 播放器异常 ({cmd[0]}): {e}")
    return False


def _speak_with_edge_tts(text):
    """使用 edge-tts 的中文女声（XiaoxiaoNeural）播报。"""
    edge_cmd = shutil.which("edge-tts")
    if not edge_cmd:
        return False

    cache_path = _edge_cache_path(text)
    if not _has_valid_audio_file(cache_path):
        ok = _generate_edge_audio_file(edge_cmd, text, cache_path, timeout_sec=_EDGE_GEN_TIMEOUT_SEC)
        if not ok:
            fail_count, backoff = _mark_edge_failure()
            print(f"[语音播报] edge-tts 失败次数={fail_count}，{backoff}秒后重试。")
            return False

    if not _play_audio_file(cache_path):
        print("[语音播报] 音频播放失败。")
        return False
    _mark_edge_success()
    return True


def _ensure_tts_worker():
    global _TTS_WORKER_STARTED, _TTS_BACKEND, _TTS_DISABLED_NOTICE_PRINTED
    with _TTS_WORKER_LOCK:
        if _TTS_WORKER_STARTED:
            return

        _TTS_BACKEND = _detect_tts_backend()
        print(f"[语音播报] 当前语音后端: {_TTS_BACKEND}")
        if _TTS_BACKEND == "edge":
            _precache_edge_phrases_async()
        elif not _TTS_DISABLED_NOTICE_PRINTED:
            print("[语音播报] 未检测到可用 edge-tts，语音播报将静默。")
            _TTS_DISABLED_NOTICE_PRINTED = True

        def worker_loop():
            global _TTS_BACKEND, _TTS_DISABLED_NOTICE_PRINTED
            while True:
                text = _TTS_QUEUE.get()
                try:
                    use_edge = _TTS_BACKEND == "edge"
                    if use_edge and _edge_available_now():
                        ok = _speak_with_edge_tts(text)
                        if not ok:
                            print("[语音播报] edge-tts 本次失败，已跳过本次播报。")
                    elif use_edge:
                        print("[语音播报] edge-tts 暂时退避中，已跳过本次播报。")
                    elif not _TTS_DISABLED_NOTICE_PRINTED:
                        print("[语音播报] 未启用可用语音后端，已跳过播报。")
                        _TTS_DISABLED_NOTICE_PRINTED = True
                except Exception as e:
                    print(f"[语音播报] 播放失败: {e}")
                finally:
                    _TTS_QUEUE.task_done()

        threading.Thread(target=worker_loop, daemon=True).start()
        _TTS_WORKER_STARTED = True


def speak_chinese_async(text, rate=138):
    """异步播报中文文本，使用单队列避免重叠抢播。"""
    del rate  # 统一由 worker 维护语速，避免多处配置导致听感波动
    normalized = _normalize_tts_text(text)
    if not normalized:
        return False

    now = time.time()
    with _TTS_DEDUP_LOCK:
        global _LAST_TTS_TEXT, _LAST_TTS_TS
        # 2 秒内的相同文案直接跳过，避免连续重复轰炸
        if normalized == _LAST_TTS_TEXT and now - _LAST_TTS_TS < 2.0:
            return True
        _LAST_TTS_TEXT = normalized
        _LAST_TTS_TS = now

    _ensure_tts_worker()
    try:
        _TTS_QUEUE.put_nowait(normalized)
    except queue.Full:
        # 队列满时丢弃最旧消息，优先播报最新提醒
        try:
            _TTS_QUEUE.get_nowait()
            _TTS_QUEUE.task_done()
        except queue.Empty:
            pass
        _TTS_QUEUE.put_nowait(normalized)
    return True

# 初始化 MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose



# ==================== 坐姿分析类 ====================
class PostureAnalyzer:
    """标准身体姿态分析模块：采用风险分数机制"""
    def __init__(self):
        self.reset_calibration() 
        
        # 姿态历史记录
        self.neck_angle_history = deque(maxlen=20)      
        self.shoulder_tilt_history = deque(maxlen=20)
        self.torso_ratio_history = deque(maxlen=20)
        self.neck_ratio_history = deque(maxlen=20)

        # 久坐提醒设置
        self.sedentary_start_time = time.time()
        self.sedentary_threshold = 45 * 60
        self.enable_sedentary = True
        self.sedentary_warning_sent = False
        
        # --- 风险分数核心参数 ---
        self.risk_score = 0.0
        self.growth_speed = 15.0  # 默认增加速度
        self.recovery_speed = 30.0 # 默认减小速度 (要求 > 增加速度)
        
        # 检测开关与灵敏度
        self.enable_head = True
        self.enable_back = True
        self.enable_shoulder = True
        self.head_sensitivity = 0.90
        self.back_sensitivity = 0.90
        self.shoulder_tolerance = 3.0
        
        self.last_speech_time = 0
        self.speech_cooldown = 5.0

    def reset_calibration(self):
        self.reference_neck_angle = None
        self.reference_shoulder_angle = None
        self.reference_torso_ratio = None
        self.reference_neck_ratio = None
        self.calibrated = False
        self.risk_score = 0.0

    def calculate_angle(self, p1, p2, p3):
        a, b, c = np.array([p1.x, p1.y]), np.array([p2.x, p2.y]), np.array([p3.x, p3.y])
        ba, bc = a - b, c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    
    def analyze_posture(self, landmarks):
        left_s, right_s = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_e, right_e = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value], landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
        left_h, right_h = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        s_mid = type('obj', (object,), {'x': (left_s.x + right_s.x) / 2, 'y': (left_s.y + right_s.y) / 2})()
        h_mid = type('obj', (object,), {'x': (left_h.x + right_h.x) / 2, 'y': (left_h.y + right_h.y) / 2})()
        e_mid = type('obj', (object,), {'x': (left_e.x + right_e.x) / 2, 'y': (left_e.y + right_e.y) / 2})()
        
        self.neck_angle_history.append(self.calculate_angle(e_mid, s_mid, h_mid))
        self.shoulder_tilt_history.append(abs(math.degrees(math.atan2(left_s.y - right_s.y, left_s.x - right_s.x))))
        
        s_width = math.sqrt((left_s.x - right_s.x)**2 + (left_s.y - right_s.y)**2)
        self.torso_ratio_history.append(abs(s_mid.y - h_mid.y) / (s_width + 1e-6))
        self.neck_ratio_history.append(abs(e_mid.y - s_mid.y) / (s_width + 1e-6))
        
        return {
            'neck_angle': np.mean(self.neck_angle_history), 'shoulder_tilt': np.mean(self.shoulder_tilt_history),
            'torso_ratio': np.mean(self.torso_ratio_history), 'neck_ratio': np.mean(self.neck_ratio_history),
            'ear_mid': e_mid, 'shoulder_mid': s_mid, 'hip_mid': h_mid
        }
    
    def calibrate(self, posture_data):
        self.reference_neck_angle = posture_data['neck_angle']
        self.reference_shoulder_angle = posture_data['shoulder_tilt']
        self.reference_torso_ratio = posture_data['torso_ratio']
        self.reference_neck_ratio = posture_data['neck_ratio']
        self.calibrated = True

    def check_posture(self, posture_data, dt):
        """核心改进：基于偏差强度计算风险分"""
        if not self.calibrated: return "未校准", [] 
        
        issues = []
        instant_risk = 0.0
        
        # 1. 计算偏差强度 (0.0 ~ 1.0)
        if self.enable_head:
            threshold = self.reference_neck_ratio * self.head_sensitivity
            if posture_data['neck_ratio'] < threshold:
                issues.append("头部前倾")
                instant_risk += (threshold - posture_data['neck_ratio']) / (threshold + 1e-6) * 10.0 # 放大权重

        if self.enable_shoulder:
            diff = abs(posture_data['shoulder_tilt'] - self.reference_shoulder_angle)
            if diff > self.shoulder_tolerance:
                issues.append("高低肩")
                instant_risk += (diff - self.shoulder_tolerance) / 5.0

        if self.enable_back:
            threshold = self.reference_torso_ratio * self.back_sensitivity
            if posture_data['torso_ratio'] < threshold:
                issues.append("含胸驼背")
                instant_risk += (threshold - posture_data['torso_ratio']) / (threshold + 1e-6) * 10.0

        # 2. 更新风险分数：采用非对称增减速逻辑
        if issues:
            # 姿态有问题：根据瞬时风险增长
            self.risk_score += (1.0 + instant_risk) * self.growth_speed * dt
        else:
            # 姿态良好：按用户设定的恢复速度快速下降
            self.risk_score -= self.recovery_speed * dt
        
        self.risk_score = max(0.0, min(100.0, self.risk_score))

        # 3. 判定状态
        current_time = time.time()
        is_sedentary = False
        if self.enable_sedentary:
            if current_time - self.sedentary_start_time > self.sedentary_threshold:
                is_sedentary = True
                issues.append("您已久坐，请起身活动！")
                if not self.sedentary_warning_sent:
                    self.last_speech_time = 0 
                    self.sedentary_warning_sent = True 
            else: self.sedentary_warning_sent = False
        
        if is_sedentary: status = "Warning"
        elif self.risk_score > 60: status = "Warning"
        elif self.risk_score > 25: status = "Attention"
        else: status = "Good"
        
        if status == "Warning" and current_time - self.last_speech_time > self.speech_cooldown:
            triggered = self.trigger_voice_alert(issues)
            if triggered:
                # ✅ 只有真正触发语音后才更新时间戳
                self.last_speech_time = current_time
            
        return status, issues

    # 功能：当 issues 为空时，根据当前风险分进行“兜底语音”
    def trigger_voice_alert(self, issues):
        """
        触发语音警报（标准身体模式）

        兜底策略：
        - Warning 区间（risk_score > 60）：播报“警告”
        - Attention 区间（risk_score > 25）：播报“注意”
        - 若存在具体 issues，优先播报具体问题
        """
        alert_text = None

        # ① 优先使用具体问题
        if issues:
            core_issues = [i.split('/')[0].strip() for i in issues if i.strip()]
            if core_issues:
                if any("久坐" in i for i in core_issues):
                    alert_text = "久坐提醒，请起身活动。"
                else:
                    issue_text = "、".join(dict.fromkeys(core_issues))
                    alert_text = f"检测到{issue_text}，请调整坐姿。"

        # ② 兜底语音（关键新增逻辑）
        if alert_text is None:
            if self.risk_score > 60:
                alert_text = "坐姿风险较高，请立即调整。"
            elif self.risk_score > 25:
                alert_text = "请注意坐姿。"
            else:
                return False  # 安全兜底：不在提醒区间

        print(f"[语音播报] {alert_text}")
        return speak_chinese_async(alert_text, rate=150)

    def reset_sedentary_timer(self):
        self.sedentary_start_time = time.time()
        self.sedentary_warning_sent = False


# ==================== 笔记本面部模式分析类 ====================
class LaptopFaceAnalyzer:
    """笔记本面部模式分析：同步改进增减速逻辑"""
    def __init__(self):
        self.SCALE_POINTS = [1, 33, 263, 234, 454, 13]
        self.UPPER_Z_POINTS = [33, 263, 70, 300]
        self.LOWER_Z_POINTS = [13, 14, 152]
        
        self.BASE_FORWARD_MAX, self.BASE_DOWN_MAX = 1.30, 0.030
        self.FORWARD_MIN, self.FORWARD_MAX = 1.05, 1.30
        self.DOWN_MIN, self.DOWN_MAX = 0.005, 0.030
        
        self.reset_calibration()
        
        # --- 用户可调参数 ---
        self.growth_speed = 15.0 
        self.recovery_speed = 30.0 
        self.last_speech_time = 0
        self.speech_cooldown = 5.0
        self.draw_scale_points = []
        self.draw_z_points = []
        
        # ✅ 久坐提醒设置（与 PostureAnalyzer 保持一致）
        self.sedentary_start_time = time.time()
        self.sedentary_threshold = 45 * 60  # 默认45分钟
        self.enable_sedentary = True
        self.sedentary_warning_sent = False

    def reset_calibration(self):
        self.baseline_scale = None
        self.baseline_z_diff = None
        self.calibrated = False
        self.risk_score = 0.0

    def analyze_face(self, landmarks, w, h, dt):
        lm = landmarks
        cx = np.mean([lm[i].x for i in self.SCALE_POINTS]) * w
        cy = np.mean([lm[i].y for i in self.SCALE_POINTS]) * h
        dists = [np.linalg.norm([lm[i].x * w - cx, lm[i].y * h - cy]) for i in self.SCALE_POINTS]
        face_scale = np.median(dists)
        z_diff = np.mean([lm[i].z for i in self.LOWER_Z_POINTS]) - np.mean([lm[i].z for i in self.UPPER_Z_POINTS])
        self.draw_scale_points = [(int(lm[i].x * w), int(lm[i].y * h)) for i in self.SCALE_POINTS]
        self.draw_z_points = [(int(lm[i].x * w), int(lm[i].y * h)) for i in self.LOWER_Z_POINTS]
        return {'face_scale': face_scale, 'z_diff': z_diff, 'dt': dt}

    def calibrate(self, data_list):
        if not data_list: return
        self.baseline_scale = np.median([d['face_scale'] for d in data_list])
        self.baseline_z_diff = np.median([d['z_diff'] for d in data_list])
        self.calibrated = True

    def check_posture(self, current_data):
        if not self.calibrated or not current_data: return "未校准", []
        
        issues = []
        dt = current_data['dt']
        scale_ratio = current_data['face_scale'] / self.baseline_scale
        z_diff_delta = current_data['z_diff'] - self.baseline_z_diff
        
        # 强度计算
        f_forward = np.clip((scale_ratio - self.FORWARD_MIN) / (self.FORWARD_MAX - self.FORWARD_MIN + 1e-6), 0, 1)
        f_down = np.clip((z_diff_delta - self.DOWN_MIN) / (self.DOWN_MAX - self.DOWN_MIN + 1e-6), 0, 1)
        instant_risk = 0.4 * f_forward + 0.6 * f_down
        
        # 风险分非对称演变
        if f_forward > 0.4 or f_down > 0.4:
            if f_forward > 0.6: issues.append("身体前倾/凑近")
            if f_down > 0.6: issues.append("低头")
            self.risk_score += (1.0 + instant_risk * 2) * self.growth_speed * dt
        else:
            self.risk_score -= self.recovery_speed * dt
            
        self.risk_score = max(0.0, min(100.0, self.risk_score))
        
        # ✅ 久坐检测逻辑（与 PostureAnalyzer 保持一致）
        current_time = time.time()
        is_sedentary = False
        if self.enable_sedentary:
            if current_time - self.sedentary_start_time > self.sedentary_threshold:
                is_sedentary = True
                issues.append("您已久坐，请起身活动！")
                if not self.sedentary_warning_sent:
                    self.last_speech_time = 0  # 强制触发语音
                    self.sedentary_warning_sent = True
            else:
                self.sedentary_warning_sent = False
        
        # 状态判定（优先久坐警告）
        if is_sedentary:
            status = "Warning"
        elif self.risk_score > 60:
            status = "Warning"
        elif self.risk_score > 30:
            status = "Attention"
        else:
            status = "Good"
            
        # 语音播报
        if status == "Warning" and (current_time - self.last_speech_time > self.speech_cooldown):
            triggered = self.trigger_voice_alert(issues)
            if triggered:
                self.last_speech_time = current_time
        
        return status, issues

    # 恢复旧版稳定实现 
    def trigger_voice_alert(self, issues):
        """
        触发语音警报（面部模式）

        兜底策略：
        - Warning（risk_score > 60）：播报“警告”
        - Attention（risk_score > 30）：播报“注意”
        """
        alert_text = None

        # ① 优先具体问题
        if issues:
            core_issues = [i.split('/')[0].strip() for i in issues if i.strip()]
            if core_issues:
                if any("久坐" in i for i in core_issues):
                    alert_text = "久坐提醒，请起身活动。"
                else:
                    issue_text = "、".join(dict.fromkeys(core_issues))
                    alert_text = f"检测到{issue_text}，请调整坐姿。"

        # ② 兜底语音
        if alert_text is None:
            if self.risk_score > 60:
                alert_text = "坐姿风险较高，请立即调整。"
            elif self.risk_score > 30:
                alert_text = "请注意坐姿。"
            else:
                return False

        print(f"[语音播报] {alert_text}")
        return speak_chinese_async(alert_text, rate=150)

    def set_forward_sensitivity(self, factor):
        self.FORWARD_MAX = self.BASE_FORWARD_MAX / (factor if factor > 0 else 1.0)
    def set_down_sensitivity(self, factor):
        self.DOWN_MAX = self.BASE_DOWN_MAX / (factor if factor > 0 else 1.0)

    # 添加久坐计时器重置方法 
    def reset_sedentary_timer(self):
        """重置久坐计时器（与 PostureAnalyzer 保持一致）"""
        self.sedentary_start_time = time.time()
        self.sedentary_warning_sent = False


# ==================== 视频处理线程 ====================
class VideoThread(QThread):
    """独立线程处理视频捕获和AI检测，完全不阻塞UI"""
    frame_ready = Signal(object)  # 发送 (frame, results, posture_data)
    
    def __init__(self, video_source=0):
        super().__init__()
        self.video_source = video_source
        self.running = False
        self.paused = False
        
        # 检测模式: 'pose' (标准身体) 或 'face' (笔记本面部)
        self.mode = 'pose' 

        # 用于标记是否需要切换摄像头
        self.pending_video_source = None 
        self.pending_mode = None # 标记模式切换
        
        # 画面处理选项
        self.rotate_180 = False
        self.mirror = True
        self.show_video = True
        
        # 检测间隔
        self.check_interval = 0.0
        self.last_check_time = 0
        
        # AI相关
        self.analyzer = PostureAnalyzer()      # 身体分析器
        self.face_analyzer = LaptopFaceAnalyzer() # 面部分析器
        self.pose = None
        self.face_mesh = None # FaceMesh 模型
        
        self.vid = None
        
        # 状态控制相关
        self.calibration_countdown = 0
        self.calibration_duration_sec = 3.0
        self.calibration_deadline = 0.0
        self.calibration_pending = False
        self.warmup_until = 0  # 用于记录预热截止时间戳

        self.calibration_data_buffer = [] # 用于存面部校准数据
        self.pose_calibration_sample = None

    # 切换模式的方法
    def change_mode(self, mode_name):
        self.pending_mode = mode_name

    # 切换摄像头的方法
    def change_source(self, source_index):
        """请求切换摄像头，实际切换在 run 循环中执行"""
        self.pending_video_source = source_index

    # 重新校准触发逻辑
    def start_calibration_countdown(self):
        """初始化校准流程：重置分析器状态并开始严格 3 秒校准窗口。"""
        if self.mode == 'pose':
            self.analyzer.reset_calibration()
            self.pose_calibration_sample = None
        else:
            self.face_analyzer.reset_calibration()
            self.calibration_data_buffer = []

        self.calibration_deadline = time.monotonic() + self.calibration_duration_sec
        self.calibration_pending = True
        self.calibration_countdown = int(math.ceil(self.calibration_duration_sec))
    
    def run(self):
        """线程主循环：集成严格的唤醒判定与超时保护"""
        self.running = True
        was_paused = False
        
        # 初始化摄像头
        self.vid = cv2.VideoCapture(self.video_source)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.init_ai_models()
        dt_last_frame = time.time()
        dt_accumulator = 0.0 

        while self.running:
            # 1. 切换逻辑
            if self.pending_mode:
                self.mode = self.pending_mode
                self.pending_mode = None
                self.init_ai_models()

            if self.pending_video_source is not None:
                if self.vid: self.vid.release()
                self.vid = cv2.VideoCapture(self.pending_video_source)
                self.pending_video_source = None
                self.warmup_until = time.time() + 2.0 # 切换源也触发预热

            # 2. 暂停逻辑
            if self.paused:
                was_paused = True
                self.msleep(100)
                continue
            
            current_time = time.time()

            # 3. 恢复后的判定逻辑
            if was_paused:
                self.warmup_until = current_time + 5.0 # 设置最长5秒唤醒等待
                dt_last_frame = current_time
                dt_accumulator = 0.0
                was_paused = False

            ret, frame = self.vid.read()
            
            # 4. 严格唤醒判定 (标准 A: 有画面且亮度达标)
            is_awake = False
            if ret and frame is not None:
                # 计算画面平均亮度，均值 > 5 视为非全黑有效画面
                if np.mean(frame) > 5:
                    is_awake = True

            # 5. 状态分发
            # 如果尚未唤醒
            if not is_awake:
                # 如果还在 5 秒宽限期内，发送唤醒中信号
                if current_time < self.warmup_until:
                    self.frame_ready.emit({
                        'frame': None, 'results': None, 'posture_data': None, 
                        'calibration_countdown': 0, 'error_msg': "CAMERA_WAKING_UP", 'mode': self.mode
                    })
                    self.msleep(100)
                    continue
                else:
                    # 超过 5 秒仍未唤醒，发送正式错误信号（触发红色 UI）
                    self.frame_ready.emit({
                        'frame': None, 'results': None, 'posture_data': None, 
                        'calibration_countdown': 0, 'error_msg': "摄像头唤醒超时或画面异常", 'mode': self.mode
                    })
                    self.msleep(500)
                    continue

            # 6. 正常运行逻辑 (已唤醒)
            if self.rotate_180: frame = cv2.rotate(frame, cv2.ROTATE_180)
            if self.mirror: frame = cv2.flip(frame, 1)
            
            dt = current_time - dt_last_frame
            dt_last_frame = current_time
            dt_accumulator += dt

            # AI 检测（仅在已唤醒状态下运行）
            results = None
            posture_data = None
            
            now_mono = time.monotonic()
            if self.calibration_pending:
                remaining = max(0.0, self.calibration_deadline - now_mono)
                self.calibration_countdown = int(math.ceil(remaining))
            else:
                self.calibration_countdown = 0

            should_run_ai = (self.check_interval <= 0.01) or \
                           (current_time - self.last_check_time > self.check_interval) or \
                           self.calibration_pending
            
            if should_run_ai:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.last_check_time = current_time
                dt_to_process = dt_accumulator
                dt_accumulator = 0.0 
                
                if self.mode == 'pose' and self.pose:
                    results = self.pose.process(img_rgb)
                    if results.pose_landmarks:
                        posture_data = self.analyzer.analyze_posture(results.pose_landmarks.landmark)
                        posture_data['dt'] = dt_to_process 
                        if self.calibration_pending and now_mono < self.calibration_deadline:
                            self.pose_calibration_sample = posture_data

                elif self.mode == 'face' and self.face_mesh:
                    results = self.face_mesh.process(img_rgb)
                    h, w, _ = frame.shape
                    if results.multi_face_landmarks:
                        face_lm = results.multi_face_landmarks[0].landmark
                        posture_data = self.face_analyzer.analyze_face(face_lm, w, h, dt_to_process)
                        if self.calibration_pending and now_mono < self.calibration_deadline:
                            self.calibration_data_buffer.append(posture_data)

            # 严格时间结束即完成校准，不再依赖帧率。
            if self.calibration_pending and time.monotonic() >= self.calibration_deadline:
                if self.mode == 'pose':
                    if self.pose_calibration_sample:
                        self.analyzer.calibrate(self.pose_calibration_sample)
                    self.pose_calibration_sample = None
                else:
                    if self.calibration_data_buffer:
                        self.face_analyzer.calibrate(self.calibration_data_buffer)
                    self.calibration_data_buffer = []
                self.calibration_pending = False
                self.calibration_countdown = 0

            # 发送数据
            self.frame_ready.emit({
                'frame': frame,
                'results': results,
                'posture_data': posture_data,
                'calibration_countdown': self.calibration_countdown,
                'error_msg': None,
                'mode': self.mode,
                'face_analyzer': self.face_analyzer if self.mode == 'face' else None
            })
            self.msleep(15)

    def init_ai_models(self):
        """根据当前模式初始化 AI 模型"""
        # 关闭旧模型
        if self.pose: self.pose.close(); self.pose = None
        if self.face_mesh: self.face_mesh.close(); self.face_mesh = None
        
        if self.mode == 'pose':
            self.pose = mp_pose.Pose(
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        elif self.mode == 'face':
            mp_face = mp.solutions.face_mesh
            self.face_mesh = mp_face.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
    
    def stop(self):
        """停止线程"""
        self.running = False
        self.wait()


# --- 在主窗口类定义前添加辅助函数 ---
class CtrlWheelSlider(QSlider):
    """仅在按住 Ctrl 时响应滚轮，避免误改参数。"""
    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            super().wheelEvent(event)
            return
        event.ignore()


def create_slider_row(label_text, min_v, max_v, def_v, callback):
    """辅助函数：创建带标签、滑块和数值显示的行布局"""
    row_layout = QHBoxLayout()
    row_layout.setContentsMargins(0, 2, 0, 2)
    row_layout.setSpacing(8)
    label = QLabel(label_text)
    label.setFixedWidth(108)
    slider = CtrlWheelSlider(Qt.Horizontal)
    slider.setMinimumWidth(120)
    slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    slider.setRange(min_v, max_v)
    slider.setValue(def_v)
    val_label = QLabel(str(def_v))
    val_label.setFixedWidth(78)
    val_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    
    slider.valueChanged.connect(lambda v: (val_label.setText(str(v)), callback(v)))
    
    row_layout.addWidget(label)
    row_layout.addWidget(slider)
    row_layout.addWidget(val_label)
    return row_layout, slider, val_label

# ==================== 主窗口 ====================
class PostureMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("坐姿矫正辅助系统 (AI Posture Assistant) - PySide6")
        self.setMinimumSize(1280, 780)
        self.video_width = 820
        self.video_height = 620
        
        # 1. 首先初始化所有需要被引用的成员变量 (Widget 实例)
        self.init_ui_components()
        self.apply_global_style()
        
        # 2. 设置布局
        central_widget = QWidget()
        central_widget.setObjectName("root")
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(14)
        
        # ==================== 左侧：展示区 (寄语 + 视频) ====================
        # 优化左侧布局结构，实现视频向上吸附
        left_container = QVBoxLayout()
        left_container.setSpacing(12)
        
        # 1. 将寄语标签加入布局 (居中对齐)
        left_container.addWidget(self.slogan_label, 0, Qt.AlignHCenter)
        
        # 2. 视频显示区配置
        self.video_label = QLabel()
        self.video_label.setObjectName("videoCanvas")
        self.video_label.setMinimumSize(self.video_width, self.video_height)
        self.video_label.setMaximumSize(self.video_width, self.video_height)
        self.video_label.setAlignment(Qt.AlignCenter)
        left_container.addWidget(self.video_label, 0, Qt.AlignHCenter)
        
        # 3. 在最下方添加一个弹簧（Stretch），将寄语和视频向上推，消除不必要的空白
        left_container.addStretch(1)
        
        main_layout.addLayout(left_container)
        
        # ==================== 右侧：控制面板 ====================
        control_panel = QWidget()
        control_panel.setObjectName("controlPanel")
        control_panel.setFixedWidth(420)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(6, 6, 6, 6)
        control_layout.setSpacing(12)
        main_layout.addWidget(control_panel)

        # A. 状态与风险监控区 (置顶)
        self.setup_monitor_section(control_layout)

        # B. 滚动配置区
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_content = QWidget()
        scroll_vbox = QVBoxLayout(scroll_content)
        scroll_vbox.setSpacing(15)

        self.setup_config_section(scroll_vbox)     # 系统配置
        self.setup_sedentary_section(scroll_vbox)  # 久坐提醒
        self.setup_algorithm_section(scroll_vbox)  # 算法参数

        scroll_vbox.addStretch(1)
        scroll_area.setWidget(scroll_content)
        control_layout.addWidget(scroll_area)

        # C. 校准按钮 (置底)
        self.calibrate_btn = QPushButton("校准坐姿 (3秒倒计时)")
        self.calibrate_btn.setObjectName("primaryButton")
        self.calibrate_btn.setMinimumHeight(50)
        self.calibrate_btn.setToolTip("按 F5 快速校准")  # 添加快捷键提示
        self.calibrate_btn.clicked.connect(self.start_calibration)
        control_layout.addWidget(self.calibrate_btn)
        
        # ==================== 初始化视频线程 ====================
        self.video_thread = VideoThread(video_source=0)
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.start()
        
        # 缓存与黑屏占位图
        self.cached_status = "Uncalibrated"
        self.cached_issues = []
        self.black_pixmap = QPixmap(self.video_width, self.video_height)
        self.black_pixmap.fill(QColor(0, 0, 0))
        self.hidden_pixmap = self.black_pixmap.copy()
        from PySide6.QtGui import QPainter
        painter = QPainter(self.hidden_pixmap)
        painter.setPen(QColor(128, 128, 128)); painter.setFont(QFont("Arial", 20))
        painter.drawText(self.hidden_pixmap.rect(), Qt.AlignCenter, "监测中 (画面已隐藏)")
        painter.end()

        self.last_error_speech_time = 0
        self.load_settings() # 加载历史配置

        # 添加 F5 快捷键支持
        from PySide6.QtGui import QKeySequence
        from PySide6.QtGui import QShortcut
        calibrate_shortcut = QShortcut(QKeySequence("F5"), self)
        calibrate_shortcut.activated.connect(self.start_calibration)

    def apply_global_style(self):
        self.setStyleSheet("""
            #root {
                background-color: #161b22;
            }
            QWidget {
                color: #e6edf3;
                font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
                font-size: 14px;
            }
            #controlPanel {
                background: #1f2630;
                border: 1px solid #2f3845;
                border-radius: 14px;
            }
            #sloganCard {
                background: #d8dee7;
                color: #2e3640;
                border: 1px solid #c6ced9;
                border-radius: 10px;
                padding: 12px 14px;
                font-size: 22px;
                font-style: italic;
                font-weight: 600;
            }
            #videoCanvas {
                background: #05070a;
                border: 2px solid #303947;
                border-radius: 12px;
            }
            QGroupBox {
                background: #222b36;
                border: 1px solid #3a4758;
                border-radius: 10px;
                margin-top: 12px;
                padding: 10px 10px 12px 10px;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #f0f6fc;
                background: #222b36;
            }
            QComboBox, QPushButton {
                min-height: 32px;
                border-radius: 8px;
                border: 1px solid #3b485a;
                background: #2b3644;
                padding: 4px 10px;
            }
            QComboBox::drop-down {
                width: 26px;
                border: none;
            }
            QComboBox QAbstractItemView {
                background: #2a333f;
                color: #e6edf3;
                border: 1px solid #3b485a;
                selection-background-color: #335d96;
            }
            QPushButton:hover {
                background: #344355;
            }
            QPushButton:checked {
                background: #4a2f2f;
                border-color: #8c4d4d;
            }
            #primaryButton {
                background: #3ca85a;
                color: #ffffff;
                border: 1px solid #5ec878;
                font-size: 17px;
                font-weight: 800;
            }
            #primaryButton:hover {
                background: #49bf67;
            }
            #modeHint {
                color: #9fb0c3;
                font-size: 12px;
            }
            #thresholdInfo {
                color: #9fb0c3;
            }
            #statusBanner {
                padding: 12px;
                border-radius: 10px;
                border: 1px solid #445366;
                background: #273240;
                color: #e6edf3;
                font-size: 17px;
                font-weight: 700;
            }
            #countdownPill {
                min-height: 36px;
                border-radius: 8px;
                background: #26384f;
                color: #dbeafe;
                border: 1px solid #3a536f;
                font-size: 22px;
                font-weight: 800;
            }
            #riskBar {
                border: 1px solid #3f4c5e;
                border-radius: 6px;
                background: #131922;
            }
            #riskBar::chunk {
                border-radius: 5px;
                background: #4CAF50;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #66768a;
                border-radius: 3px;
                background: #1a202a;
            }
            QCheckBox::indicator:checked {
                background: #52a96b;
                border-color: #6ed78c;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #17202b;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #6d77ff;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
                background: #cad4ff;
                border: 1px solid #9ca9e3;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)

    def _set_status_banner(self, text, state="neutral"):
        styles = {
            "neutral": ("#273240", "#445366", "#e6edf3"),
            "info": ("#2a3441", "#4d6078", "#dbeafe"),
            "good": ("#1f3a2b", "#2f6b46", "#c7f9d6"),
            "attention": ("#463721", "#866430", "#ffe8ad"),
            "warning": ("#482327", "#a74856", "#ffd8de"),
            "error": ("#5c232b", "#c84f5d", "#ffe1e6"),
            "paused": ("#313943", "#5a687a", "#d2d9e2"),
        }
        bg, border, fg = styles.get(state, styles["neutral"])
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"padding: 12px; border-radius: 10px; background-color: {bg}; "
            f"border: 1px solid {border}; color: {fg}; font-size: 17px; font-weight: 700;"
        )

    def _set_countdown_style(self, state="normal"):
        styles = {
            "normal": ("#26384f", "#3a536f", "#dbeafe"),
            "soon": ("#4a3c23", "#8f7240", "#ffe7bb"),
            "alert": ("#5d252d", "#cc4f5e", "#ffe2e7"),
            "off": ("#2f3540", "#5f6979", "#c2c9d3"),
        }
        bg, border, fg = styles.get(state, styles["normal"])
        self.countdown_label.setStyleSheet(
            f"background-color: {bg}; color: {fg}; border: 1px solid {border}; "
            "font-size: 22px; font-weight: 800; padding: 6px; border-radius: 8px;"
        )

    def _set_risk_bar_color(self, color):
        self.risk_bar.setStyleSheet(
            "QProgressBar#riskBar { border: 1px solid #3f4c5e; border-radius: 6px; background: #131922; }"
            f"QProgressBar#riskBar::chunk {{ background: {color}; border-radius: 5px; }}"
        )

    def init_ui_components(self):
        """预先实例化所有关键 UI 组件，防止引用错误"""
        self.status_label = QLabel("状态: 未校准")
        self.status_label.setObjectName("statusBanner")
        self.risk_bar = QProgressBar()
        self.risk_bar.setObjectName("riskBar")
        self.risk_score_label = QLabel("风险分: 0")
        self.risk_score_label.setFont(QFont("Consolas", 14, QFont.Bold))
        self.threshold_info = QLabel("阈值: 注意>25 | 警告>60")
        self.threshold_info.setObjectName("thresholdInfo")
        
        self.countdown_label = QLabel("久坐倒计时: --:--")
        self.countdown_label.setObjectName("countdownPill")
        self.enable_sedentary_check = QCheckBox("启用久坐提醒")
        self.enable_sedentary_check.setChecked(True)
        self.enable_sedentary_check.stateChanged.connect(self.on_sedentary_toggle)
        self.reset_timer_btn = QPushButton("重置计时")
        self.reset_timer_btn.clicked.connect(self.reset_sedentary_timer)
        self.test_tts_btn = QPushButton("测试语音播报")
        self.test_tts_btn.clicked.connect(self.on_test_tts_clicked)
        
        self.body_settings_widget = QWidget()
        self.face_settings_widget = QWidget()
        self.face_settings_widget.setVisible(False)

        # 摄像头开关
        self.mirror_check = QCheckBox("水平镜像")
        self.show_video_check = QCheckBox("实时画面")
        self.rotate_180_check = QCheckBox("旋转180°") 

        # 每日寄语组件初始化
        self.slogan_label = QLabel()
        self.slogan_label.setObjectName("sloganCard")
        self.slogan_label.setAlignment(Qt.AlignCenter)
        self.slogan_label.setWordWrap(True)
        # 固定宽度以对齐视频，设置最小高度保证美观
        self.slogan_label.setFixedWidth(self.video_width)
        self.slogan_label.setMinimumHeight(60)
        self.slogan_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.current_slogan = ""
        self.last_slogan_date = ""

    def setup_monitor_section(self, layout):
        # 状态标签
        self.status_label.setMinimumHeight(65)
        self.status_label.setAlignment(Qt.AlignCenter)
        self._set_status_banner("状态: 未校准 (请保持端正后点击校准)", "neutral")
        layout.addWidget(self.status_label)

        # 风险条
        risk_group = QGroupBox("实时风险监控")
        risk_l = QVBoxLayout(risk_group)
        self.risk_bar.setFixedHeight(12)
        self.risk_bar.setTextVisible(False)
        risk_l.addWidget(self.risk_bar)
        
        info_l = QHBoxLayout()
        self.risk_score_label.setFont(QFont("Consolas", 14, QFont.Bold))
        info_l.addWidget(self.risk_score_label)
        info_l.addStretch()
        info_l.addWidget(self.threshold_info)
        risk_l.addLayout(info_l)
        layout.addWidget(risk_group)

    def setup_config_section(self, layout):
        group = QGroupBox("系统配置")
        l = QVBoxLayout(group)
        
        l.addWidget(QLabel("摄像头源:"))
        self.cam_combo = QComboBox()
        self.cam_combo.addItems(["摄像头 0", "摄像头 1", "摄像头 2"])
        self.cam_combo.currentIndexChanged.connect(self.on_cam_source_changed)
        l.addWidget(self.cam_combo)

        l.addWidget(QLabel("场景模式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["标准模式 (检测上半身)", "专注模式 (仅检测面部)"])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        l.addWidget(self.mode_combo)
        
        self.mode_hint = QLabel("适合：全身/半身入镜")
        self.mode_hint.setObjectName("modeHint")
        l.addWidget(self.mode_hint)

        c_l = QHBoxLayout()
        self.mirror_check.setChecked(True)
        self.mirror_check.stateChanged.connect(self.on_mirror_changed)
        self.show_video_check.setChecked(True)
        self.show_video_check.stateChanged.connect(self.on_show_video_changed)
        self.rotate_180_check.stateChanged.connect(self.on_rotate_changed)
        c_l.addWidget(self.mirror_check); c_l.addWidget(self.show_video_check); c_l.addWidget(self.rotate_180_check)
        l.addLayout(c_l)
        
        self.pause_btn = QPushButton("暂停检测")
        self.pause_btn.setCheckable(True)
        self.pause_btn.clicked.connect(self.toggle_pause)
        l.addWidget(self.pause_btn)

        l.addWidget(self.test_tts_btn)
        layout.addWidget(group)

    def setup_sedentary_section(self, layout):
        group = QGroupBox("久坐提醒")
        l = QVBoxLayout(group)
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self._set_countdown_style("normal")
        l.addWidget(self.countdown_label)
        
        ctrl_l = QHBoxLayout()
        ctrl_l.addWidget(self.enable_sedentary_check)
        ctrl_l.addStretch()
        ctrl_l.addWidget(self.reset_timer_btn)
        l.addLayout(ctrl_l)
        
        row, self.sedentary_slider, self.sedentary_label = create_slider_row("提醒周期(分):", 1, 120, 45, self.on_sedentary_changed)
        l.addLayout(row)
        layout.addWidget(group)

    def setup_algorithm_section(self, layout):
        group = QGroupBox("算法灵敏度调整")
        vbox = QVBoxLayout(group)
        
        r1, self.growth_slider, self.growth_label = create_slider_row("风险增长:", 5, 50, 15, self.on_risk_speed_changed)
        r2, self.recover_slider, self.recover_label = create_slider_row("风险恢复:", 5, 100, 30, self.on_risk_speed_changed)
        r3, self.interval_slider, self.interval_label = create_slider_row("AI间隔:", 0, 50, 0, self.on_interval_changed)
        vbox.addLayout(r1); vbox.addLayout(r2); vbox.addLayout(r3)

        # 身体模式细节
        body_l = QVBoxLayout(self.body_settings_widget)
        b1, self.enable_head_check, self.head_slider, self.head_label = self.create_check_slider_block("头前倾", 50, 110, 90, self.on_head_sens_changed)
        b2, self.enable_back_check, self.back_slider, self.back_label = self.create_check_slider_block("驼背", 50, 110, 90, self.on_back_sens_changed)
        b3, self.enable_shoulder_check, self.shoulder_slider, self.shoulder_label = self.create_check_slider_block("高低肩(°)", 5, 150, 30, self.on_shoulder_sens_changed)
        body_l.addLayout(b1); body_l.addLayout(b2); body_l.addLayout(b3)
        vbox.addWidget(self.body_settings_widget)

        # 面部模式细节
        face_l = QVBoxLayout(self.face_settings_widget)
        f1, self.face_forward_slider, self.face_forward_label = create_slider_row("前倾灵敏度:", 50, 150, 100, self.on_face_forward_changed)
        f2, self.face_down_slider, self.face_down_label = create_slider_row("低头灵敏度:", 50, 150, 100, self.on_face_down_changed)
        face_l.addLayout(f1); face_l.addLayout(f2)
        vbox.addWidget(self.face_settings_widget)
        
        layout.addWidget(group)

    def create_check_slider_block(self, name, min_v, max_v, def_v, callback):
        """内部工具：创建带复选框的滑动条块"""
        l = QVBoxLayout()
        chk = QCheckBox(name); chk.setChecked(True); chk.stateChanged.connect(self.on_toggle_changed)
        row, sld, lbl = create_slider_row("灵敏度:", min_v, max_v, def_v, callback)
        l.addWidget(chk); l.addLayout(row)
        return l, chk, sld, lbl
    
    # ==================== 回调函数 ====================
    def on_mode_changed(self, index):
        # 切换模式时重置 UI 层的状态缓存，防止文本闪烁或残留
        self.cached_status = "Uncalibrated"
        self.cached_issues = []
        
        # 切换模式时更新 UI 阈值提示 
        if index == 0:
            self.video_thread.change_mode('pose')
            self.mode_hint.setText("适合：全身/半身入镜，检测驼背/高低肩")
            self.body_settings_widget.setVisible(True)
            self.face_settings_widget.setVisible(False)
            self.threshold_info.setText("阈值: 注意>25 | 警告>60") # 标准模式阈值
        else:
            self.video_thread.change_mode('face')
            self.mode_hint.setText("适合：笔记本近距离，只露出头部，检测前倾/低头")
            self.body_settings_widget.setVisible(False)
            self.face_settings_widget.setVisible(True)
            self.threshold_info.setText("阈值: 注意>30 | 警告>60") # 面部模式阈值
        
        self._set_status_banner("状态: 模式切换中，请重新校准", "info")

    # 面部模式参数回调
    def on_face_forward_changed(self, value):
        factor = value / 100.0
        self.video_thread.face_analyzer.set_forward_sensitivity(factor)
        self.face_forward_label.setText(f"系数: {factor:.2f}")
        
    def on_face_down_changed(self, value):
        factor = value / 100.0
        self.video_thread.face_analyzer.set_down_sensitivity(factor)
        self.face_down_label.setText(f"系数: {factor:.2f}")

    # 摄像头源切换回调
    def on_cam_source_changed(self, index):
        print(f"切换摄像头源至: {index}")
        self.video_thread.change_source(index)

    def on_rotate_changed(self, state):
        # 直接通过复选框对象获取布尔值状态
        self.video_thread.rotate_180 = self.rotate_180_check.isChecked()
    
    def on_mirror_changed(self, state):
        # 直接通过复选框对象获取布尔值状态
        self.video_thread.mirror = self.mirror_check.isChecked()
    
    def on_show_video_changed(self, state):
        # 使用 isChecked() 直接获取布尔值，避免 PySide6 枚举类型比较失败的问题
        self.video_thread.show_video = self.show_video_check.isChecked()
    
    def on_toggle_changed(self):
        self.video_thread.analyzer.enable_head = self.enable_head_check.isChecked()
        self.video_thread.analyzer.enable_back = self.enable_back_check.isChecked()
        self.video_thread.analyzer.enable_shoulder = self.enable_shoulder_check.isChecked()
    
    def on_head_sens_changed(self, value):
        factor = value / 100.0
        self.video_thread.analyzer.head_sensitivity = factor
        self.head_label.setText(f"系数: {factor:.2f}")
    
    def on_back_sens_changed(self, value):
        factor = value / 100.0
        self.video_thread.analyzer.back_sensitivity = factor
        self.back_label.setText(f"系数: {factor:.2f}")
    
    def on_shoulder_sens_changed(self, value):
        degree = value / 10.0
        self.video_thread.analyzer.shoulder_tolerance = degree
        self.shoulder_label.setText(f"容差: {degree:.1f}°")
    
    def on_risk_speed_changed(self, _):
        g_val = self.growth_slider.value()
        r_val = self.recover_slider.value()
        
        # 更新标准模式参数
        self.video_thread.analyzer.growth_speed = float(g_val)
        self.video_thread.analyzer.recovery_speed = float(r_val)
        
        # 更新面部模式参数
        self.video_thread.face_analyzer.growth_speed = float(g_val)
        self.video_thread.face_analyzer.recovery_speed = float(r_val)
        
        self.growth_label.setText(f"{g_val}")
        self.recover_label.setText(f"{r_val}")

    def on_interval_changed(self, value):
        seconds = value / 10.0
        self.video_thread.check_interval = seconds
        if seconds < 0.1:
            self.interval_label.setText("间隔: 实时检测")
        else:
            self.interval_label.setText(f"间隔: {seconds:.1f}秒")
    
    def start_calibration(self):
        """开始或重新校准流程"""
        # 如果处于暂停状态，先自动恢复检测
        if self.pause_btn.isChecked():
            self.toggle_pause()  # 自动恢复检测
        
        # 调用线程中的集成重置与校准方法
        self.video_thread.start_calibration_countdown()
        
        # 更新UI显示
        self._set_status_banner("状态: 校准采集中，请保持端正姿势...", "attention")


    def toggle_pause(self):
        is_paused = self.pause_btn.isChecked()
        self.video_thread.paused = is_paused
        if is_paused:
            self.pause_btn.setText("继续检测")
            # 立即清空画面为黑屏
            self.video_label.setPixmap(self.black_pixmap)
            self._set_status_banner("状态: 已暂停", "paused")
        else:
            self.pause_btn.setText("暂停检测")
            # 恢复时先显示黑屏，等待线程唤醒
            self.video_label.setPixmap(self.black_pixmap)
            # 恢复时重置两个分析器的久坐计时
            self.video_thread.analyzer.reset_sedentary_timer()
            self.video_thread.face_analyzer.reset_sedentary_timer()

    # 久坐开关回调逻辑 
    def on_sedentary_toggle(self, state):
        # 使用 isChecked() 获取准确的布尔值
        is_enabled = self.enable_sedentary_check.isChecked()
        
        # ✅ 同时更新两个分析器的开关
        self.video_thread.analyzer.enable_sedentary = is_enabled
        self.video_thread.face_analyzer.enable_sedentary = is_enabled  
        
        # 如果重新开启，建议重置计时器
        if is_enabled:
            self.video_thread.analyzer.reset_sedentary_timer()
            self.video_thread.face_analyzer.reset_sedentary_timer()

    def on_sedentary_changed(self, value):
        self.sedentary_label.setText(f"{value}分钟")
        
        # ✅ 同时更新两个分析器的阈值
        threshold_seconds = value * 60
        self.video_thread.analyzer.sedentary_threshold = threshold_seconds
        self.video_thread.face_analyzer.sedentary_threshold = threshold_seconds
        
    def reset_sedentary_timer(self):
        # ✅ 同时重置两个分析器的计时器
        self.video_thread.analyzer.reset_sedentary_timer()
        self.video_thread.face_analyzer.reset_sedentary_timer()  
        self._set_status_banner("状态: 久坐计时已重置", "info")

    def on_test_tts_clicked(self):
        self._set_status_banner("状态: 正在测试语音播报", "info")
        speak_chinese_async("语音播报测试成功")
    
    def update_frame(self, data):
        """更新视频帧及 UI 状态"""
        error_msg = data.get('error_msg')
        if error_msg:
            if error_msg == "CAMERA_WAKING_UP":
                # 唤醒中：保持黑屏，仅更新状态标签
                self._set_status_banner("状态: 正在唤醒摄像头...", "info")
                self.video_label.setPixmap(self.black_pixmap)
                return

            # 真实的异常处理：红色提示界面
            self.video_label.setPixmap(self.hidden_pixmap)
            self._set_status_banner(error_msg, "error")
            
            # 异常语音播报 (每30秒一次)
            current_time = time.time()
            if current_time - self.last_error_speech_time > 30.0:
                self.last_error_speech_time = current_time
                speak_chinese_async("摄像头连接异常，请检查摄像头是否可用。")
            return
        
        frame = data['frame']
        results = data['results']
        posture_data = data['posture_data']
        calibration_countdown = data['calibration_countdown']
        current_mode = data.get('mode', 'pose')

        # 2. 久坐倒计时逻辑
        if self.video_thread.analyzer.enable_sedentary:
            elapsed_time = time.time() - self.video_thread.analyzer.sedentary_start_time
            remaining_seconds = self.video_thread.analyzer.sedentary_threshold - elapsed_time
            if remaining_seconds <= 0:
                self.countdown_label.setText("该起身活动了")
                self._set_countdown_style("alert")
            else:
                mins, secs = divmod(int(remaining_seconds), 60)
                self.countdown_label.setText(f"久坐倒计时: {mins:02d}:{secs:02d}")
                self._set_countdown_style("soon" if remaining_seconds < 300 else "normal")
        else:
            self.countdown_label.setText("久坐提醒: 已关闭")
            self._set_countdown_style("off")
        
        # 3. 状态检查与缓存逻辑
        current_status = self.cached_status
        current_issues = self.cached_issues
        is_calibrated = False

        if current_mode == 'pose':
            if self.video_thread.analyzer.calibrated:
                is_calibrated = True
                if posture_data:
                    current_status, current_issues = self.video_thread.analyzer.check_posture(posture_data, posture_data.get('dt', 0.033))
                    self.cached_status, self.cached_issues = current_status, current_issues
        elif current_mode == 'face':
            if self.video_thread.face_analyzer.calibrated:
                is_calibrated = True
                if posture_data:
                    current_status, current_issues = self.video_thread.face_analyzer.check_posture(posture_data)
                    self.cached_status, self.cached_issues = current_status, current_issues

        # 4. 更新风险条与状态标签
        # ✅ 优先判断：如果正在校准中，保持校准提示，不更新其他状态
        if calibration_countdown > 0:
            # 校准进行中：保持显示校准提示文本和样式
            self._set_status_banner("状态: 校准采集中，请保持端正姿势...", "attention")
        elif is_calibrated:
            # 已校准：正常显示检测状态
            analyzer = self.video_thread.analyzer if current_mode == 'pose' else self.video_thread.face_analyzer
            score = int(analyzer.risk_score)
            self.risk_score_label.setText(f"风险分: {score}")
            self.risk_bar.setValue(score)
            
            att_val = 25 if current_mode == 'pose' else 30
            color = "#f44336" if score >= 60 else ("#ff9800" if score >= att_val else "#4CAF50")
            self._set_risk_bar_color(color)

            if current_status == "Good":
                self._set_status_banner("状态: 坐姿良好", "good")
            elif current_status == "Attention":
                self._set_status_banner("状态: 注意坐姿", "attention")
            elif current_status == "Warning":
                self._set_status_banner(f"警告: {', '.join(current_issues)}", "warning")
        else:
            # 未校准状态
            self._set_status_banner("状态: 未校准 (请保持端正后点击校准)", "neutral")

        # 5. 绘图与显示
        if not self.video_thread.show_video:
            self.video_label.setPixmap(self.hidden_pixmap)
            return

        if current_mode == 'pose' and results and results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        elif current_mode == 'face' and results and results.multi_face_landmarks:
            face_analyzer = data.get('face_analyzer')
            if face_analyzer:
                for (px, py) in face_analyzer.draw_scale_points: cv2.circle(frame, (px, py), 3, (255, 0, 0), -1)
                for (px, py) in face_analyzer.draw_z_points: cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)

        if calibration_countdown > 0:
            text = f"Calibrating... {calibration_countdown}"
            cv2.putText(frame, text, (frame.shape[1]//2 - 150, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        self.display_frame(frame)

    def display_frame(self, frame):
        """辅助方法：将 OpenCV 帧显示到标签上"""
        if frame is None: return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qt_image = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(qt_image).scaled(
                self.video_width, self.video_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    # 添加保存/加载设置的方法 
    def save_settings(self):
        """保存当前用户设置及寄语状态到 JSON 文件"""
        settings = {
            # 摄像头设置
            "rotate_180": self.rotate_180_check.isChecked(),
            "mirror": self.mirror_check.isChecked(),
            "show_video": self.show_video_check.isChecked(),
            
            # 检测参数
            "growth_speed": self.growth_slider.value(),
            "recovery_speed": self.recover_slider.value(),
            "check_interval": self.interval_slider.value(),
            
            # 开关设置
            "enable_head": self.enable_head_check.isChecked(),
            "enable_back": self.enable_back_check.isChecked(),
            "enable_shoulder": self.enable_shoulder_check.isChecked(),
            
            # 灵敏度设置
            "head_sensitivity": self.head_slider.value(),
            "back_sensitivity": self.back_slider.value(),
            "shoulder_tolerance": self.shoulder_slider.value(),
            "face_forward_sens": self.face_forward_slider.value(),
            "face_down_sens": self.face_down_slider.value(),
            
            # 久坐设置
            "enable_sedentary": self.enable_sedentary_check.isChecked(),
            "sedentary_minutes": self.sedentary_slider.value(),

            # 寄语持久化
            "last_slogan_date": self.last_slogan_date,
            "current_slogan": self.current_slogan
        }
        
        try:
            with open("user_settings.json", "w", encoding='utf-8') as f:
                json.dump(settings, f, indent=4, ensure_ascii=False)
            print("设置已保存")
        except Exception as e:
            print(f"保存设置失败: {e}")

    def load_settings(self):
        """从 JSON 文件加载用户设置，并处理每日寄语逻辑"""
        settings_path = "user_settings.json"
        if not os.path.exists(settings_path):
            # 若无配置文件，仍尝试加载今日寄语
            self._update_slogan_logic({})
            return
            
        try:
            with open(settings_path, "r", encoding='utf-8') as f:
                settings = json.load(f)
            
            # 摄像头设置
            if "rotate_180" in settings: self.rotate_180_check.setChecked(settings["rotate_180"])
            if "mirror" in settings: self.mirror_check.setChecked(settings["mirror"])
            if "show_video" in settings: self.show_video_check.setChecked(settings["show_video"])
            
            # 风险机制参数
            if "growth_speed" in settings: self.growth_slider.setValue(settings["growth_speed"])
            if "recovery_speed" in settings: self.recover_slider.setValue(settings["recovery_speed"])
            if "check_interval" in settings: self.interval_slider.setValue(settings["check_interval"])
            
            # 开关设置
            if "enable_head" in settings: self.enable_head_check.setChecked(settings["enable_head"])
            if "enable_back" in settings: self.enable_back_check.setChecked(settings["enable_back"])
            if "enable_shoulder" in settings: self.enable_shoulder_check.setChecked(settings["enable_shoulder"])
            
            # 灵敏度设置
            if "head_sensitivity" in settings: self.head_slider.setValue(settings["head_sensitivity"])
            if "back_sensitivity" in settings: self.back_slider.setValue(settings["back_sensitivity"])
            if "shoulder_tolerance" in settings: self.shoulder_slider.setValue(settings["shoulder_tolerance"])
            if "face_forward_sens" in settings: self.face_forward_slider.setValue(settings["face_forward_sens"])
            if "face_down_sens" in settings: self.face_down_slider.setValue(settings["face_down_sens"])
            
            # 久坐设置
            if "enable_sedentary" in settings: self.enable_sedentary_check.setChecked(settings["enable_sedentary"])
            if "sedentary_minutes" in settings: self.sedentary_slider.setValue(settings["sedentary_minutes"])
            
            # 执行寄语更新逻辑
            self._update_slogan_logic(settings)
            
            print("设置与寄语加载完成")
        except Exception as e:
            print(f"加载设置失败: {e}")
            self.slogan_label.hide()
    
    def _update_slogan_logic(self, settings):
        """核心逻辑：根据日期决定显示旧寄语还是抽取新寄语"""
        today_str = str(date.today())
        stored_date = settings.get("last_slogan_date", "")
        stored_slogan = settings.get("current_slogan", "")

        # 如果日期不匹配或记录为空，则重新抽取
        if stored_date != today_str or not stored_slogan:
            self.current_slogan = self._fetch_new_slogan()
            self.last_slogan_date = today_str
        else:
            self.current_slogan = stored_slogan
            self.last_slogan_date = stored_date

        # UI 展示处理
        if self.current_slogan:
            self.slogan_label.setText(f"“ {self.current_slogan} ”")
            self.slogan_label.show()
        else:
            self.slogan_label.hide()

    def _fetch_new_slogan(self):
        """从文件中解析并随机获取一条寄语"""
        file_path = "./data/slogans.txt"
        if not os.path.exists(file_path):
            print(f"寄语文件未找到: {file_path}")
            return None
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                # 过滤出以 "- " 开头的行，并去除前缀和空格
                candidates = [
                    line.strip()[2:].strip() 
                    for line in f 
                    if line.strip().startswith("- ") and len(line.strip()) > 2
                ]
            
            if candidates:
                return random.choice(candidates)
        except Exception as e:
            print(f"读取寄语文件出错: {e}")
        
        return None

    def closeEvent(self, event):
        """窗口关闭时优雅停止线程"""
        # 关闭前保存设置 
        self.save_settings()

        print("正在关闭...")
        self.video_thread.stop()
        event.accept()


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle("Fusion")
    
    # 创建主窗口
    window = PostureMainWindow()
    window.show()
    
    sys.exit(app.exec())

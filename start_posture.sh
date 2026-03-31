#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/postcorr_env"
APP_FILE="${SCRIPT_DIR}/posture_pyside6.py"
DEFAULT_TTS_PROXY="${DEFAULT_TTS_PROXY:-http://127.0.0.1:7890}"
DEFAULT_TTS_BACKEND="${DEFAULT_TTS_BACKEND:-edge}"

if [[ ! -f "${APP_FILE}" ]]; then
  echo "未找到启动文件: ${APP_FILE}"
  exit 1
fi

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "未找到虚拟环境: ${VENV_PATH}"
  exit 1
fi

source "${VENV_PATH}/bin/activate"
cd "${SCRIPT_DIR}"

# 语音代理设置：优先使用外部已配置代理；否则回落到本地常见代理端口。
# 启动前确保代理已启用
if [[ -z "${POSTURE_TTS_PROXY:-}" ]]; then
  POSTURE_TTS_PROXY="${HTTPS_PROXY:-${https_proxy:-${HTTP_PROXY:-${http_proxy:-${DEFAULT_TTS_PROXY}}}}}"
fi
export POSTURE_TTS_PROXY
echo "TTS代理: ${POSTURE_TTS_PROXY}"

# 语音后端设置：默认强制 edge，可通过外部环境变量覆盖。
export POSTURE_TTS_BACKEND="${POSTURE_TTS_BACKEND:-${DEFAULT_TTS_BACKEND}}"
echo "TTS后端: ${POSTURE_TTS_BACKEND}"

exec python "${APP_FILE}" "$@"

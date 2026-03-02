"""
项目配置模块
从 .env 文件加载阿里云百炼 API 配置
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# ======================== API 配置 ========================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# OpenAI 兼容接口 Base URL
OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ======================== 模型配置 ========================
# 图像理解模型
IMAGE_UNDERSTANDING_MODEL = "qwen-vl-max"

# 音频理解模型
AUDIO_UNDERSTANDING_MODEL = "qwen3-omni-flash"

# 图像生成模型
IMAGE_GENERATION_MODEL = "wanx2.1-t2i-turbo"

# 语音合成模型 (CosyVoice)
SPEECH_SYNTHESIS_MODEL = "cosyvoice-v1"
# 语音合成 - 发音人 (可选: longxiaochun, longhua, longshuo 等)
SPEECH_SYNTHESIS_VOICE = "longxiaochun"

# 语音合成模型 (Qwen-TTS) — 支持 base64 返回
QWEN_TTS_MODEL = "qwen3-tts-flash"
# Qwen-TTS 发音人 (可选: Cherry, Ethan, Serena, Chelsie 等)
QWEN_TTS_VOICE = "Cherry"

# ======================== 路径配置 ========================
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

# 确保目录存在
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def validate_api_key():
    """校验 API Key 是否已配置"""
    if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "your_api_key_here":
        raise ValueError(
            "请先配置 DASHSCOPE_API_KEY！\n"
            "1. 复制 .env.example 为 .env\n"
            "2. 在 .env 中填入你的 API Key\n"
            "3. API Key 获取地址: https://bailian.console.aliyun.com/"
        )

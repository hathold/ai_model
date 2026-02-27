"""
语音合成服务
使用 CosyVoice 模型将文本转换为语音
"""
import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer
from pathlib import Path

import config
from services.api_logger import log_api_call


def synthesize_speech(
    text: str,
    output_file: str = None,
    voice: str = None,
    model: str = None,
) -> str:
    """
    语音合成：将文本转换为语音并保存为音频文件

    Args:
        text: 要合成的文本内容
        output_file: 输出文件路径，默认保存到 output 目录
        voice: 发音人，默认使用配置中的发音人
        model: 模型名称，默认使用配置中的模型

    Returns:
        生成的音频文件路径
    """
    config.validate_api_key()
    dashscope.api_key = config.DASHSCOPE_API_KEY
    model = model or config.SPEECH_SYNTHESIS_MODEL
    voice = voice or config.SPEECH_SYNTHESIS_VOICE

    if output_file is None:
        output_file = str(config.OUTPUT_DIR / "speech_output.mp3")

    synthesizer = SpeechSynthesizer(
        model=model,
        voice=voice,
    )

    request_params = {
        "model": model,
        "voice": voice,
        "text": text,
    }

    audio_data = synthesizer.call(text)

    # 记录日志（响应体为音频二进制，记录元信息）
    response_info = {
        "status": "success",
        "audio_size_bytes": len(audio_data) if audio_data else 0,
        "output_file": output_file,
    }
    log_api_call("speech_synthesis", model, request_params, response_info)

    # 保存音频文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(audio_data)

    print(f"语音已保存: {output_path}")
    return str(output_path)


def synthesize_speech_stream(
    text: str,
    output_file: str = None,
    voice: str = None,
    model: str = None,
) -> str:
    """
    语音合成（流式）：逐步生成语音并写入文件

    Args:
        text: 要合成的文本内容
        output_file: 输出文件路径
        voice: 发音人
        model: 模型名称

    Returns:
        生成的音频文件路径
    """
    config.validate_api_key()
    dashscope.api_key = config.DASHSCOPE_API_KEY
    model = model or config.SPEECH_SYNTHESIS_MODEL
    voice = voice or config.SPEECH_SYNTHESIS_VOICE

    if output_file is None:
        output_file = str(config.OUTPUT_DIR / "speech_output_stream.mp3")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 使用回调的方式流式写入
    class AudioCallback:
        def __init__(self, file_path):
            self.file = open(file_path, "wb")

        def on_data(self, data: bytes):
            self.file.write(data)

        def close(self):
            self.file.close()

    callback = AudioCallback(output_path)

    synthesizer = SpeechSynthesizer(
        model=model,
        voice=voice,
        callback=callback,
    )

    synthesizer.streaming_call(text)
    synthesizer.streaming_complete()
    callback.close()

    print(f"语音已保存: {output_path}")
    return str(output_path)


# ===================== 快捷测试 =====================
if __name__ == "__main__":
    print("=" * 50)
    print("语音合成测试")
    print("=" * 50)
    path = synthesize_speech(
        text="你好，我是通义千问语音合成服务。今天天气真不错！",
    )
    print(f"生成语音: {path}")

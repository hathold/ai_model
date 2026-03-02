"""
语音合成服务 (Qwen-TTS)
使用千问 TTS 模型将文本转换为语音
支持两种返回方式：
  - 非流式 (stream=False)：返回音频文件 URL
  - 流式   (stream=True) ：返回 base64 编码的音频数据
"""
import base64
from pathlib import Path

import dashscope

import config
from services.api_logger import log_api_call


def synthesize_speech_qwen(
    text: str,
    output_file: str = None,
    voice: str = None,
    model: str = None,
) -> dict:
    """
    Qwen-TTS 语音合成（非流式）：返回音频文件 URL

    Args:
        text: 要合成的文本内容
        output_file: 如指定，则将音频下载并保存到此路径
        voice: 发音人，默认使用配置中的发音人
        model: 模型名称，默认使用配置中的模型

    Returns:
        dict: 包含 url 等信息的字典
    """
    config.validate_api_key()
    dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
    model = model or config.QWEN_TTS_MODEL
    voice = voice or config.QWEN_TTS_VOICE

    request_params = {
        "model": model,
        "voice": voice,
        "text": text,
        "stream": False,
    }

    response = dashscope.MultiModalConversation.call(
        api_key=config.DASHSCOPE_API_KEY,
        model=model,
        text=text,
        voice=voice,
        stream=False,
    )

    log_api_call("speech_synthesis_qwen", model, request_params, response)

    audio_url = None
    if hasattr(response, "output") and hasattr(response.output, "audio"):
        audio_url = response.output.audio.get("url") if isinstance(response.output.audio, dict) else getattr(response.output.audio, "url", None)
    elif isinstance(response, dict):
        audio_url = response.get("output", {}).get("audio", {}).get("url")

    result = {
        "url": audio_url,
        "request_id": getattr(response, "request_id", None),
    }

    # 如果指定了 output_file，下载音频并保存
    if output_file and audio_url:
        import urllib.request
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(audio_url, str(output_path))
        result["output_file"] = str(output_path)
        print(f"语音已保存: {output_path}")

    return result


def synthesize_speech_qwen_base64(
    text: str,
    output_file: str = None,
    voice: str = None,
    model: str = None,
) -> dict:
    """
    Qwen-TTS 语音合成（流式 / base64）：接口直接返回 base64 编码的音频数据

    Args:
        text: 要合成的文本内容
        output_file: 如指定，则将音频解码后保存到此路径
        voice: 发音人，默认使用配置中的发音人
        model: 模型名称，默认使用配置中的模型

    Returns:
        dict: 包含 audio_base64、audio_size_bytes 等信息
    """
    config.validate_api_key()
    dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
    model = model or config.QWEN_TTS_MODEL
    voice = voice or config.QWEN_TTS_VOICE

    request_params = {
        "model": model,
        "voice": voice,
        "text": text,
        "stream": True,
    }

    # 流式调用，逐块收集 base64 音频片段
    responses = dashscope.MultiModalConversation.call(
        api_key=config.DASHSCOPE_API_KEY,
        model=model,
        text=text,
        voice=voice,
        stream=True,
    )

    audio_base64_chunks = []
    last_response = None

    for response in responses:
        last_response = response
        audio_data_b64 = None

        # 提取每个 chunk 中的 base64 音频数据
        if hasattr(response, "output") and hasattr(response.output, "audio"):
            audio_obj = response.output.audio
            if isinstance(audio_obj, dict):
                audio_data_b64 = audio_obj.get("data")
            else:
                audio_data_b64 = getattr(audio_obj, "data", None)
        elif isinstance(response, dict):
            audio_data_b64 = response.get("output", {}).get("audio", {}).get("data")

        if audio_data_b64:
            audio_base64_chunks.append(audio_data_b64)

    # 拼接所有 base64 片段
    full_audio_base64 = "".join(audio_base64_chunks)

    # 解码以获取字节大小
    audio_bytes = base64.b64decode(full_audio_base64) if full_audio_base64 else b""

    response_info = {
        "status": "success",
        "format": "base64",
        "audio_base64_length": len(full_audio_base64),
        "audio_size_bytes": len(audio_bytes),
        "chunks_count": len(audio_base64_chunks),
    }
    log_api_call("speech_synthesis_qwen/base64", model, request_params, response_info)

    result = {
        "audio_base64": full_audio_base64,
        "audio_size_bytes": len(audio_bytes),
        "chunks_count": len(audio_base64_chunks),
    }

    # 如果指定了 output_file，解码并保存
    if output_file and audio_bytes:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        result["output_file"] = str(output_path)
        print(f"语音已保存: {output_path}")

    print(f"语音合成完成 (base64), 音频大小: {len(audio_bytes)} bytes, 分片数: {len(audio_base64_chunks)}")
    return result


# ===================== 快捷测试 =====================
if __name__ == "__main__":
    print("=" * 50)
    print("Qwen-TTS 语音合成测试")
    print("=" * 50)

    test_text = "你好，我是千问语音合成服务。今天天气真不错！"

    print("\n--- 非流式（返回 URL）---")
    result1 = synthesize_speech_qwen(text=test_text)
    print(f"音频 URL: {result1.get('url')}")

    print("\n--- 流式（返回 base64）---")
    result2 = synthesize_speech_qwen_base64(
        text=test_text,
        output_file=str(config.OUTPUT_DIR / "qwen_tts_base64_output.wav"),
    )
    print(f"base64 长度: {len(result2.get('audio_base64', ''))}")
    print(f"音频大小: {result2.get('audio_size_bytes')} bytes")

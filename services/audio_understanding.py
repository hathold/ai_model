"""
音频理解服务
使用 Qwen-Audio 模型对音频进行理解和分析
支持本地音频和网络音频
支持 DashScope 原生格式 和 OpenAI 兼容格式 两种调用方式
"""
import dashscope
from dashscope import MultiModalConversation
from openai import OpenAI

import config
from services.api_logger import log_api_call, log_stream_api_call


# ======================== DashScope 格式 ========================


def understand_audio(
    audio_path: str,
    prompt: str = "请描述这段音频的内容。",
    model: str = None,
) -> str:
    """
    音频理解（DashScope 格式）：分析音频内容并返回描述

    Args:
        audio_path: 音频路径（本地路径或 URL）
        prompt: 提问内容
        model: 模型名称，默认使用配置中的模型

    Returns:
        模型对音频的理解描述
    """
    config.validate_api_key()
    dashscope.api_key = config.DASHSCOPE_API_KEY
    model = model or config.AUDIO_UNDERSTANDING_MODEL

    # 判断是本地文件还是 URL
    if audio_path.startswith(("http://", "https://")):
        audio_content = {"audio": audio_path}
    else:
        audio_content = {"audio": f"file://{audio_path}"}

    messages = [
        {
            "role": "user",
            "content": [
                audio_content,
                {"text": prompt},
            ],
        }
    ]

    request_params = {"model": model, "messages": messages}

    response = MultiModalConversation.call(
        model=model,
        messages=messages,
    )

    log_api_call("audio_understanding/dashscope", model, request_params, response)

    if response.status_code == 200:
        return response.output.choices[0].message.content[0]["text"]
    else:
        raise Exception(
            f"音频理解调用失败: code={response.code}, message={response.message}"
        )


# ======================== OpenAI 兼容格式 ========================


def understand_audio_openai(
    audio_path: str,
    prompt: str = "请描述这段音频的内容。",
    model: str = None,
) -> str:
    """
    音频理解（OpenAI 兼容格式）：分析音频内容并返回描述

    Args:
        audio_path: 音频路径（本地路径或 URL）
        prompt: 提问内容
        model: 模型名称

    Returns:
        模型对音频的理解描述
    """
    config.validate_api_key()
    model = model or config.AUDIO_UNDERSTANDING_MODEL

    client = OpenAI(
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.OPENAI_BASE_URL,
    )

    # OpenAI 兼容格式的音频内容
    if audio_path.startswith(("http://", "https://")):
        audio_url = audio_path
    else:
        audio_url = f"file://{audio_path}"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_url, "format": "mp3"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    request_params = {"model": model, "messages": messages}

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    log_api_call("audio_understanding/openai", model, request_params, response)

    return response.choices[0].message.content


# ===================== 快捷测试 =====================
if __name__ == "__main__":
    test_url = "https://dashscope.oss-cn-beijing.aliyuncs.com/audios/welcome.mp3"

    print("=" * 60)
    print("音频理解测试 — DashScope 格式")
    print("=" * 60)
    result = understand_audio(test_url, "这段音频说了什么？")
    print(result)

    print("\n" + "=" * 60)
    print("音频理解测试 — OpenAI 兼容格式")
    print("=" * 60)
    result = understand_audio_openai(test_url, "这段音频说了什么？")
    print(result)

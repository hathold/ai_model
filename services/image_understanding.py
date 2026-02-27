"""
图像理解服务
使用 Qwen-VL 模型对图像进行理解和描述
支持本地图片和网络图片（URL / 本地路径 / base64）
支持 DashScope 原生格式 和 OpenAI 兼容格式 两种调用方式
"""
import base64
import mimetypes

import dashscope
from dashscope import MultiModalConversation
from openai import OpenAI

import config
from services.api_logger import log_api_call, log_stream_api_call


def _image_to_base64(image_path: str) -> tuple[str, str]:
    """
    将本地图片文件读取并转为 base64 编码字符串

    Returns:
        (base64_str, mime_type) 例如 ("iVBOR...", "image/png")
    """
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/png"  # 默认
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime_type


# ======================== DashScope 格式 ========================


def understand_image(
    image_path: str,
    prompt: str = "请详细描述这张图片的内容。",
    model: str = None,
) -> str:
    """
    图像理解（DashScope 格式）：分析图片内容并返回描述

    Args:
        image_path: 图片路径（本地路径或 URL）
        prompt: 提问内容
        model: 模型名称，默认使用配置中的模型

    Returns:
        模型对图片的理解描述
    """
    config.validate_api_key()
    dashscope.api_key = config.DASHSCOPE_API_KEY
    model = model or config.IMAGE_UNDERSTANDING_MODEL

    # 判断是本地文件还是 URL
    if image_path.startswith(("http://", "https://")):
        image_content = {"image": image_path}
    else:
        image_content = {"image": f"file://{image_path}"}

    messages = [
        {
            "role": "user",
            "content": [
                image_content,
                {"text": prompt},
            ],
        }
    ]

    request_params = {"model": model, "messages": messages}

    response = MultiModalConversation.call(
        model=model,
        messages=messages,
    )

    log_api_call("image_understanding/dashscope", model, request_params, response)

    if response.status_code == 200:
        return response.output.choices[0].message.content[0]["text"]
    else:
        raise Exception(
            f"图像理解调用失败: code={response.code}, message={response.message}"
        )


def understand_image_base64(
    image_path: str,
    prompt: str = "请详细描述这张图片的内容。",
    model: str = None,
) -> str:
    """
    图像理解（DashScope 格式 + base64）：以 base64 编码传递图片

    报文结构示例::

        {
            "model": "qwen-vl-max",
            "messages": [{
                "role": "user",
                "content": [
                    {"image": "data:image/png;base64,iVBORw0KGgo..."},
                    {"text": "请详细描述这张图片的内容。"}
                ]
            }]
        }

    Args:
        image_path: 本地图片路径
        prompt: 提问内容
        model: 模型名称

    Returns:
        模型对图片的理解描述
    """
    config.validate_api_key()
    dashscope.api_key = config.DASHSCOPE_API_KEY
    model = model or config.IMAGE_UNDERSTANDING_MODEL

    base64_str, mime_type = _image_to_base64(image_path)
    image_data_url = f"data:{mime_type};base64,{base64_str}"

    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_data_url},
                {"text": prompt},
            ],
        }
    ]

    request_params = {"model": model, "messages": messages}

    response = MultiModalConversation.call(
        model=model,
        messages=messages,
    )

    log_api_call("image_understanding/dashscope_base64", model, request_params, response)

    if response.status_code == 200:
        return response.output.choices[0].message.content[0]["text"]
    else:
        raise Exception(
            f"图像理解调用失败: code={response.code}, message={response.message}"
        )


# ======================== OpenAI 兼容格式 ========================


def understand_image_openai(
    image_path: str,
    prompt: str = "请详细描述这张图片的内容。",
    model: str = None,
) -> str:
    """
    图像理解（OpenAI 兼容格式）：分析图片内容并返回描述

    Args:
        image_path: 图片路径（本地路径或 URL）
        prompt: 提问内容
        model: 模型名称

    Returns:
        模型对图片的理解描述
    """
    config.validate_api_key()
    model = model or config.IMAGE_UNDERSTANDING_MODEL

    client = OpenAI(
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.OPENAI_BASE_URL,
    )

    # OpenAI 格式的图片内容
    if image_path.startswith(("http://", "https://")):
        image_url = image_path
    else:
        image_url = f"file://{image_path}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    request_params = {"model": model, "messages": messages}

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    log_api_call("image_understanding/openai", model, request_params, response)

    return response.choices[0].message.content


def understand_image_openai_base64(
    image_path: str,
    prompt: str = "请详细描述这张图片的内容。",
    model: str = None,
) -> str:
    """
    图像理解（OpenAI 兼容格式 + base64）：以 base64 编码传递图片

    报文结构示例::

        {
            "model": "qwen-vl-max",
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgo..."
                        }
                    },
                    {"type": "text", "text": "请详细描述这张图片的内容。"}
                ]
            }]
        }

    Args:
        image_path: 本地图片路径
        prompt: 提问内容
        model: 模型名称

    Returns:
        模型对图片的理解描述
    """
    config.validate_api_key()
    model = model or config.IMAGE_UNDERSTANDING_MODEL

    client = OpenAI(
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.OPENAI_BASE_URL,
    )

    base64_str, mime_type = _image_to_base64(image_path)
    image_data_url = f"data:{mime_type};base64,{base64_str}"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url},
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

    log_api_call("image_understanding/openai_base64", model, request_params, response)

    return response.choices[0].message.content


# ===================== 快捷测试 =====================
if __name__ == "__main__":
    test_url = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"

    print("=" * 60)
    print("图像理解测试 — DashScope 格式")
    print("=" * 60)
    result = understand_image(test_url, "这张图片里有什么？")
    print(result)

    print("\n" + "=" * 60)
    print("图像理解测试 — OpenAI 兼容格式")
    print("=" * 60)
    result = understand_image_openai(test_url, "这张图片里有什么？")
    print(result)

    # base64 格式测试（需要本地图片）
    import os
    test_local = os.path.join(config.INPUT_DIR, "cat.png")
    if os.path.exists(test_local):
        print("\n" + "=" * 60)
        print("图像理解测试 — DashScope base64 格式")
        print("=" * 60)
        result = understand_image_base64(test_local, "这张图片里有什么？")
        print(result)

        print("\n" + "=" * 60)
        print("图像理解测试 — OpenAI base64 格式")
        print("=" * 60)
        result = understand_image_openai_base64(test_local, "这张图片里有什么？")
        print(result)
    else:
        print(f"\n跳过 base64 测试: 未找到本地测试图片 {test_local}")

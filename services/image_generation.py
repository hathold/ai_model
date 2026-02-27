"""
图像生成服务
使用通义万相模型根据文本描述生成图像
"""
import dashscope
from dashscope import ImageSynthesis
from pathlib import Path

import config
from services.api_logger import log_api_call


def generate_image(
    prompt: str,
    negative_prompt: str = "",
    size: str = "1024*1024",
    n: int = 1,
    model: str = None,
    save_dir: str = None,
) -> list[str]:
    """
    图像生成：根据文本描述生成图像

    Args:
        prompt: 图像描述（正向提示词）
        negative_prompt: 负面提示词（不希望出现的内容）
        size: 图像尺寸，如 "1024*1024", "720*1280", "1280*720"
        n: 生成图片数量（1-4）
        model: 模型名称，默认使用配置中的模型
        save_dir: 保存目录，默认使用配置中的输出目录

    Returns:
        生成图片的本地保存路径列表
    """
    config.validate_api_key()
    dashscope.api_key = config.DASHSCOPE_API_KEY
    model = model or config.IMAGE_GENERATION_MODEL
    save_dir = Path(save_dir) if save_dir else config.OUTPUT_DIR

    request_params = {
        "model": model,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "n": n,
        "size": size,
    }

    response = ImageSynthesis.call(
        model=model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        n=n,
        size=size,
    )

    log_api_call("image_generation", model, request_params, response)

    if response.status_code == 200:
        saved_paths = []
        for i, result in enumerate(response.output.results):
            # 下载并保存图片
            import urllib.request

            file_name = f"generated_{i + 1}.png"
            file_path = save_dir / file_name
            urllib.request.urlretrieve(result.url, str(file_path))
            saved_paths.append(str(file_path))
            print(f"图片已保存: {file_path}")

        return saved_paths
    else:
        raise Exception(
            f"图像生成调用失败: code={response.code}, message={response.message}"
        )


def generate_image_url(
    prompt: str,
    negative_prompt: str = "",
    size: str = "1024*1024",
    n: int = 1,
    model: str = None,
) -> list[str]:
    """
    图像生成：返回生成图片的 URL（不下载到本地）

    Args:
        prompt: 图像描述
        negative_prompt: 负面提示词
        size: 图像尺寸
        n: 生成数量
        model: 模型名称

    Returns:
        生成图片的 URL 列表
    """
    config.validate_api_key()
    dashscope.api_key = config.DASHSCOPE_API_KEY
    model = model or config.IMAGE_GENERATION_MODEL

    request_params = {
        "model": model,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "n": n,
        "size": size,
    }

    response = ImageSynthesis.call(
        model=model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        n=n,
        size=size,
    )

    log_api_call("image_generation", model, request_params, response)

    if response.status_code == 200:
        return [result.url for result in response.output.results]
    else:
        raise Exception(
            f"图像生成调用失败: code={response.code}, message={response.message}"
        )


# ===================== 快捷测试 =====================
if __name__ == "__main__":
    print("=" * 50)
    print("图像生成测试")
    print("=" * 50)
    paths = generate_image(
        prompt="一只可爱的橘猫坐在窗台上，阳光洒在身上，水彩画风格",
        n=1,
        size="1024*1024",
    )
    for p in paths:
        print(f"生成图片: {p}")

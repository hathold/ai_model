"""
阿里云百炼 多模态模型功能测试
============================
支持功能：
  1. 图像理解 (Qwen-VL)       — DashScope / OpenAI 双格式
  2. 音频理解 (Qwen-Audio)     — DashScope / OpenAI 双格式
  3. 图像生成 (通义万相)        — 仅 DashScope
  4. 语音合成 (CosyVoice)      — 仅 DashScope
"""
from services.image_understanding import (
    understand_image,
    understand_image_openai,
    understand_image_base64,
    understand_image_openai_base64,
)
from services.audio_understanding import (
    understand_audio,
    understand_audio_openai,
    understand_audio_base64,
    understand_audio_openai_base64,
)
from services.image_generation import generate_image
from services.speech_synthesis import synthesize_speech


# ===================== 图像理解 =====================

def demo_image_understanding_dashscope():
    """图像理解 — DashScope 格式"""
    print("\n" + "=" * 60)
    print("📷  图像理解 — DashScope 原生格式")
    print("=" * 60)
    image_url = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"
    result = understand_image(image_url, "请详细描述这张图片的内容。")
    print(f"\n图片: {image_url}")
    print(f"描述: {result}")


def demo_image_understanding_openai():
    """图像理解 — OpenAI 兼容格式"""
    print("\n" + "=" * 60)
    print("📷  图像理解 — OpenAI 兼容格式")
    print("=" * 60)
    image_url = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"
    result = understand_image_openai(image_url, "请详细描述这张图片的内容。")
    print(f"\n图片: {image_url}")
    print(f"描述: {result}")


def demo_image_understanding_dashscope_base64():
    """图像理解 — DashScope base64 格式"""
    print("\n" + "=" * 60)
    print("📷  图像理解 — DashScope base64 格式")
    print("=" * 60)
    import os, config
    image_path = input("请输入本地图片路径 (默认 input/test.png): ").strip()
    if not image_path:
        image_path = os.path.join(config.INPUT_DIR, "test.png")
    if not os.path.exists(image_path):
        print(f"❌ 文件不存在: {image_path}")
        return
    result = understand_image_base64(image_path, "请详细描述这张图片的内容。")
    print(f"\n图片: {image_path}")
    print(f"描述: {result}")


def demo_image_understanding_openai_base64():
    """图像理解 — OpenAI base64 格式"""
    print("\n" + "=" * 60)
    print("📷  图像理解 — OpenAI base64 格式")
    print("=" * 60)
    import os, config
    image_path = input("请输入本地图片路径 (默认 input/test.png): ").strip()
    if not image_path:
        image_path = os.path.join(config.INPUT_DIR, "test.png")
    if not os.path.exists(image_path):
        print(f"❌ 文件不存在: {image_path}")
        return
    result = understand_image_openai_base64(image_path, "请详细描述这张图片的内容。")
    print(f"\n图片: {image_path}")
    print(f"描述: {result}")


# ===================== 音频理解 =====================

def demo_audio_understanding_dashscope():
    """音频理解 — DashScope 格式"""
    print("\n" + "=" * 60)
    print("🎵  音频理解 — DashScope 原生格式")
    print("=" * 60)
    audio_url = "https://dashscope.oss-cn-beijing.aliyuncs.com/audios/welcome.mp3"
    result = understand_audio(audio_url, "这段音频说了什么？")
    print(f"\n音频: {audio_url}")
    print(f"描述: {result}")


def demo_audio_understanding_openai():
    """音频理解 — OpenAI 兼容格式"""
    print("\n" + "=" * 60)
    print("🎵  音频理解 — OpenAI 兼容格式")
    print("=" * 60)
    audio_url = "https://dashscope.oss-cn-beijing.aliyuncs.com/audios/welcome.mp3"
    result = understand_audio_openai(audio_url, "这段音频说了什么？")
    print(f"\n音频: {audio_url}")
    print(f"描述: {result}")


def demo_audio_understanding_dashscope_base64():
    """音频理解 — DashScope base64 格式"""
    print("\n" + "=" * 60)
    print("🎵  音频理解 — DashScope base64 格式")
    print("=" * 60)
    import os, config
    audio_path = input("请输入本地音频路径 (默认 input/test.mp3): ").strip()
    if not audio_path:
        audio_path = os.path.join(config.INPUT_DIR, "test.mp3")
    if not os.path.exists(audio_path):
        print(f"❌ 文件不存在: {audio_path}")
        return
    result = understand_audio_base64(audio_path, "这段音频说了什么？")
    print(f"\n音频: {audio_path}")
    print(f"描述: {result}")


def demo_audio_understanding_openai_base64():
    """音频理解 — OpenAI base64 格式"""
    print("\n" + "=" * 60)
    print("🎵  音频理解 — OpenAI base64 格式")
    print("=" * 60)
    import os, config
    audio_path = input("请输入本地音频路径 (默认 input/test.mp3): ").strip()
    if not audio_path:
        audio_path = os.path.join(config.INPUT_DIR, "test.mp3")
    if not os.path.exists(audio_path):
        print(f"❌ 文件不存在: {audio_path}")
        return
    result = understand_audio_openai_base64(audio_path, "这段音频说了什么？")
    print(f"\n音频: {audio_path}")
    print(f"描述: {result}")


# ===================== 图像生成 & 语音合成 =====================

def demo_image_generation():
    """图像生成（仅 DashScope）"""
    print("\n" + "=" * 60)
    print("🎨  图像生成 — DashScope（无 OpenAI 兼容接口）")
    print("=" * 60)
    prompt = "一只可爱的橘猫坐在窗台上，阳光洒在身上，水彩画风格"
    paths = generate_image(prompt=prompt, n=1)
    print(f"\n提示词: {prompt}")
    for p in paths:
        print(f"生成图片: {p}")


def demo_speech_synthesis():
    """语音合成（仅 DashScope）"""
    print("\n" + "=" * 60)
    print("🔊  语音合成 — DashScope（无 OpenAI 兼容接口）")
    print("=" * 60)
    text = "你好，我是通义千问语音合成服务。今天天气真不错，一起出去走走吧！"
    path = synthesize_speech(text=text)
    print(f"\n文本: {text}")
    print(f"语音文件: {path}")


def main():
    print("🚀 阿里云百炼 - 多模态模型功能测试")
    print("=" * 60)
    print("请选择要测试的功能：")
    print()
    print("  --- 图像理解 (支持双格式对比) ---")
    print("  1a. 图像理解 — DashScope 格式")
    print("  1b. 图像理解 — OpenAI 兼容格式")
    print("  1c. 图像理解 — 两种格式对比运行")
    print("  1d. 图像理解 — DashScope base64 格式")
    print("  1e. 图像理解 — OpenAI base64 格式")
    print("  1f. 图像理解 — base64 两种格式对比运行")
    print()
    print("  --- 音频理解 (支持双格式对比) ---")
    print("  2a. 音频理解 — DashScope 格式")
    print("  2b. 音频理解 — OpenAI 兼容格式")
    print("  2c. 音频理解 — 两种格式对比运行")
    print("  2d. 音频理解 — DashScope base64 格式")
    print("  2e. 音频理解 — OpenAI base64 格式")
    print("  2f. 音频理解 — base64 两种格式对比运行")
    print()
    print("  --- 仅 DashScope 格式 ---")
    print("  3.  图像生成")
    print("  4.  语音合成")
    print()
    print("  5.  运行全部")
    print("  0.  退出")
    print("=" * 60)

    demos = {
        "1a": ("图像理解(DashScope)", demo_image_understanding_dashscope),
        "1b": ("图像理解(OpenAI)", demo_image_understanding_openai),
        "1d": ("图像理解(DashScope-base64)", demo_image_understanding_dashscope_base64),
        "1e": ("图像理解(OpenAI-base64)", demo_image_understanding_openai_base64),
        "2a": ("音频理解(DashScope)", demo_audio_understanding_dashscope),
        "2b": ("音频理解(OpenAI)", demo_audio_understanding_openai),
        "2d": ("音频理解(DashScope-base64)", demo_audio_understanding_dashscope_base64),
        "2e": ("音频理解(OpenAI-base64)", demo_audio_understanding_openai_base64),
        "3": ("图像生成", demo_image_generation),
        "4": ("语音合成", demo_speech_synthesis),
    }

    compare_groups = {
        "1c": [("图像理解(DashScope)", demo_image_understanding_dashscope),
               ("图像理解(OpenAI)", demo_image_understanding_openai)],
        "1f": [("图像理解(DashScope-base64)", demo_image_understanding_dashscope_base64),
               ("图像理解(OpenAI-base64)", demo_image_understanding_openai_base64)],
        "2c": [("音频理解(DashScope)", demo_audio_understanding_dashscope),
               ("音频理解(OpenAI)", demo_audio_understanding_openai)],
        "2f": [("音频理解(DashScope-base64)", demo_audio_understanding_dashscope_base64),
               ("音频理解(OpenAI-base64)", demo_audio_understanding_openai_base64)],
    }

    while True:
        choice = input("\n请输入选项: ").strip().lower()

        if choice == "0":
            print("再见！")
            break
        elif choice == "5":
            for name, func in demos.values():
                try:
                    func()
                except Exception as e:
                    print(f"\n❌ {name} 失败: {e}")
        elif choice in compare_groups:
            print("\n🔄 对比运行两种格式，日志将分别保存，便于对比报文结构...")
            for name, func in compare_groups[choice]:
                try:
                    func()
                except Exception as e:
                    print(f"\n❌ {name} 失败: {e}")
            print("\n✅ 对比完成！请查看 logs/ 目录下对应子目录的 JSON 文件。")
        elif choice in demos:
            name, func = demos[choice]
            try:
                func()
            except Exception as e:
                print(f"\n❌ {name} 失败: {e}")
        else:
            print("无效选项，请重新输入。")


if __name__ == "__main__":
    main()

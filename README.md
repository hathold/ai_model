# 阿里云百炼 - 多模态模型功能测试

基于阿里云百炼平台 DashScope SDK，调用多模态大模型 API。

## 功能列表

| 功能 | 模型 | 说明 |
|------|------|------|
| 图像理解 | Qwen-VL-Max | 分析图片内容，回答关于图片的问题 |
| 音频理解 | Qwen-Audio-Turbo | 分析音频内容，进行语音识别和理解 |
| 图像生成 | 通义万相 (Wanx) | 根据文本描述生成图像 |
| 语音合成 | CosyVoice | 将文本转换为自然语音 |

## 快速开始

### 1. 创建虚拟环境

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # macOS / Linux
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
copy .env.example .env
```

编辑 `.env` 文件，填入你的百炼 API Key：

```
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
```

> API Key 获取地址：https://bailian.console.aliyun.com/

### 4. 运行

```bash
python main.py
```

## 项目结构

```
ai_model/
├── .env.example          # 环境变量模板
├── .gitignore
├── README.md
├── requirements.txt      # Python 依赖
├── config.py             # 配置管理
├── main.py               # 主入口（交互式菜单）
├── services/
│   ├── __init__.py
│   ├── image_understanding.py   # 图像理解
│   ├── audio_understanding.py   # 音频理解
│   ├── image_generation.py      # 图像生成
│   └── speech_synthesis.py      # 语音合成
├── input/                # 输入文件目录
└── output/               # 输出文件目录
```

## 单独测试某个功能

```bash
# 图像理解
python -m services.image_understanding

# 音频理解
python -m services.audio_understanding

# 图像生成
python -m services.image_generation

# 语音合成
python -m services.speech_synthesis
```

## 在代码中使用

```python
from services.image_understanding import understand_image
from services.audio_understanding import understand_audio
from services.image_generation import generate_image
from services.speech_synthesis import synthesize_speech

# 图像理解
result = understand_image("https://example.com/image.jpg", "图片里有什么？")

# 音频理解
result = understand_audio("path/to/audio.mp3", "这段音频说了什么？")

# 图像生成
paths = generate_image("一只猫在草地上玩耍", n=1)

# 语音合成
audio_path = synthesize_speech("你好，世界！")
```

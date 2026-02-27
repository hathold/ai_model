"""
API 请求/响应日志记录模块
将每次 API 调用的请求体和响应体保存为 JSON 文件，方便查看报文结构
"""
import json
import time
from datetime import datetime
from pathlib import Path

import config

# 日志保存目录
LOG_DIR = config.BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


def _make_serializable(obj):
    """将对象转换为可 JSON 序列化的格式"""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, bytes):
        return f"<bytes: {len(obj)} bytes>"
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    # dashscope 响应对象 — 尝试常见属性
    if hasattr(obj, "__dict__"):
        return _make_serializable(vars(obj))
    return str(obj)


def _response_to_dict(response) -> dict:
    """将 DashScope 响应对象转为字典"""
    # OpenAI SDK 响应对象 — 有 model_dump_json / model_dump 方法
    if hasattr(response, "model_dump"):
        return response.model_dump()
    # DashScope SDK 响应对象
    result = {}
    for attr in ("status_code", "code", "message", "request_id", "output", "usage"):
        if hasattr(response, attr):
            result[attr] = _make_serializable(getattr(response, attr))
    return result


def log_api_call(
    service: str,
    model: str,
    request_params: dict,
    response,
    extra: dict = None,
) -> str:
    """
    记录一次 API 调用的请求体和响应体

    Args:
        service: 服务名称，如 "image_understanding"
        model: 模型名称
        request_params: 请求参数字典
        response: DashScope 响应对象（或字典）
        extra: 附加信息

    Returns:
        日志文件路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_ms = int(time.time() * 1000) % 10000  # 毫秒后缀防重名

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "service": service,
        "model": model,
        "request": _make_serializable(request_params),
        "response": (
            _response_to_dict(response)
            if not isinstance(response, dict)
            else response
        ),
    }

    if extra:
        log_entry["extra"] = _make_serializable(extra)

    # 按服务分子目录
    service_dir = LOG_DIR / service
    service_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{timestamp}_{ts_ms}.json"
    file_path = service_dir / file_name

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, ensure_ascii=False, indent=2)

    print(f"📝 日志已保存: {file_path}")
    return str(file_path)


def log_stream_api_call(
    service: str,
    model: str,
    request_params: dict,
    responses: list,
    extra: dict = None,
) -> str:
    """
    记录流式 API 调用的请求体和所有响应片段

    Args:
        service: 服务名称
        model: 模型名称
        request_params: 请求参数字典
        responses: 所有流式响应片段的列表
        extra: 附加信息

    Returns:
        日志文件路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_ms = int(time.time() * 1000) % 10000

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "service": service,
        "model": model,
        "stream": True,
        "request": _make_serializable(request_params),
        "responses": [
            _response_to_dict(r) if not isinstance(r, dict) else r
            for r in responses
        ],
    }

    if extra:
        log_entry["extra"] = _make_serializable(extra)

    service_dir = LOG_DIR / service
    service_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{timestamp}_{ts_ms}_stream.json"
    file_path = service_dir / file_name

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, ensure_ascii=False, indent=2)

    print(f"📝 日志已保存: {file_path}")
    return str(file_path)

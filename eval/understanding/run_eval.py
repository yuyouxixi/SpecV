#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multimodal Understanding checklist (SVP) evaluation.

Sends each model output together with the task prompt and the per-item checklist to a judge model
(default: gemini-3-flash-preview, via the DashScope OpenAI-compatible endpoint) and records a Yes/No
answer for every checklist item. Supports multiprocessing, resume (skips finished items) and retries.
Output mirrors the checklist with an added "answer" field, under results/<model>/<dimension>/<id>.json.
Run score_and_rank.py afterwards to aggregate per-dimension and overall scores.

Usage:
    # one model
    python run_eval.py --model-name nano_banana --num-processes 50
    # several models
    python run_eval.py --model-name nano_banana claude_opus_4.5 --num-processes 50
    # quick test (a few items)
    python run_eval.py --model-name nano_banana --num-processes 2 --limit 5
    # pick a different judge model
    python run_eval.py --model-name nano_banana --judge-model gpt-4.1

Environment variables:
    DASHSCOPE_API_KEY   (required) judge-model API key
    JUDGE_API_URL       (required) judge API endpoint
    JUDGE_MODEL         (required) judge model id (SpecV used Gemini 3 Flash; --judge-model overrides)
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import time
import random
import fcntl
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from multiprocessing import Pool
from datetime import datetime

try:
    from PIL import Image
except ImportError:
    print(
        "[ERROR] Pillow 未安装，请执行: pip install Pillow",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)


# ============== 路径配置 ==============
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
CHECKLIST_DIR = PROJECT_DIR / "eval_checklist" / "understanding"
MODEL_OUTPUT_DIR = PROJECT_DIR / "model_outputs" / "understanding"
DATA_DIR = PROJECT_DIR / "eval_data" / "understanding"
PROMPT_FILE = BASE_DIR / "eval_prompt.txt"
RESULTS_DIR = BASE_DIR / "results"

# ============== API 配置 ==============
API_URL = os.environ.get("JUDGE_API_URL", "")

# ============== 裁判模型配置 ==============
# SpecV was evaluated with Gemini 3 Flash as the judge. The judge model id and the base URL
# depend on your provider — supply them yourself (the request format is unchanged):
#   JUDGE_API_URL   judge API endpoint (see API 配置 above)
#   JUDGE_MODEL     the model id your provider exposes (e.g. a Gemini 3 Flash id)
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "")

_JUDGE_MODEL_NAME = JUDGE_MODEL
_JUDGE_MODEL_TYPE = "gemini"

# ============== 默认配置 ==============
DEFAULT_MAX_RETRIES = 6
DEFAULT_TIMEOUT = 180
DEFAULT_NUM_PROCESSES = 100
DEFAULT_TEMPERATURE = 0.1
WORKER_MAX_RETRIES = 3

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 400, 408, 409, 423, 449}

# ============== 图片压缩配置 ==============
MAX_IMAGE_SIZE_BYTES = 2 * 1024 * 1024  # 2 MB
INITIAL_JPEG_QUALITY = 95  # 初始转换 JPEG 的质量


# ============== 维度自动检测 ==============
def detect_dimensions(checklist_dir: Path) -> List[str]:
    """
    自动检测 checklist 目录下的子目录名作为维度列表。
    """
    if not checklist_dir.exists():
        print(f"[ERROR] Checklist 目录不存在: {checklist_dir}", file=sys.stderr, flush=True)
        sys.exit(1)

    dimensions = sorted([d.name for d in checklist_dir.iterdir() if d.is_dir()])
    if not dimensions:
        print(f"[ERROR] Checklist 目录下没有子目录: {checklist_dir}", file=sys.stderr, flush=True)
        sys.exit(1)

    return dimensions


# ============== JSON 解析工具 ==============
def _fix_unescaped_quotes(json_str: str, max_fixes: int = 50):
    fixed = json_str
    for _ in range(max_fixes):
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            pos = e.pos
            found = False
            for i in range(pos - 1, max(0, pos - 100), -1):
                if fixed[i] == '"' and (i == 0 or fixed[i - 1] != '\\'):
                    test = fixed[:i] + '\\"' + fixed[i + 1:]
                    try:
                        return json.loads(test)
                    except json.JSONDecodeError as e2:
                        if e2.pos > e.pos:
                            fixed = test
                            found = True
                            break
            if not found:
                for i in range(pos, min(len(fixed), pos + 10)):
                    if fixed[i] == '"' and (i == 0 or fixed[i - 1] != '\\'):
                        test = fixed[:i] + '\\"' + fixed[i + 1:]
                        try:
                            return json.loads(test)
                        except json.JSONDecodeError as e2:
                            if e2.pos > e.pos:
                                fixed = test
                                found = True
                                break
            if not found:
                return None
    return None


def _fix_missing_opening_quotes(json_str: str) -> str:
    fixed = re.sub(
        r'(:\s*)(?!["{\[\d\-]|true\b|false\b|null\b)(?=\S)',
        r'\1"',
        json_str,
    )
    fixed = re.sub(
        r'(:\s*)\[(?=[^\s"\[\]{}\d\]tf\-n])',
        r'\1"[',
        fixed,
    )
    return fixed


def parse_json_response(content: str):
    """
    从模型响应中解析 JSON（支持列表和字典格式）
    """
    if not content:
        return None

    content_fixed = _fix_missing_opening_quotes(content)

    for text in (content, content_fixed):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    try:
        match = re.search(r'```(?:json|JSON)?\s*\n?(.*?)```', content, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            return json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        pass

    try:
        match = re.search(r'```(?:json|JSON)?\s*\n?(.*?)```', content, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            json_str_fixed = _fix_missing_opening_quotes(json_str)
            return json.loads(json_str_fixed)
    except (json.JSONDecodeError, IndexError):
        pass

    try:
        match = re.search(r'```(?:json|JSON)?\s*\n?(.*?)```', content, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            result = _fix_unescaped_quotes(json_str)
            if result is not None:
                return result
    except (IndexError, Exception):
        pass

    try:
        match = re.search(r'```(?:json|JSON)?\s*\n?(.*?)```', content, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            json_str_fixed = _fix_missing_opening_quotes(json_str)
            result = _fix_unescaped_quotes(json_str_fixed)
            if result is not None:
                return result
    except (IndexError, Exception):
        pass

    try:
        match = re.search(r'```(?:json|JSON)?\s*\n?(.*?)```', content, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            return json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        pass

    try:
        if "```json" in content or "```JSON" in content:
            idx = content.find("```json")
            if idx == -1:
                idx = content.find("```JSON")
            json_str = content[idx + 7:].strip()
            json_str = json_str.rstrip('`').strip()
            return json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        pass

    try:
        if "```json" in content or "```JSON" in content:
            idx = content.find("```json")
            if idx == -1:
                idx = content.find("```JSON")
            json_str = content[idx + 7:].strip()
            json_str = json_str.rstrip('`').strip()
            json_str_fixed = _fix_missing_opening_quotes(json_str)
            try:
                return json.loads(json_str_fixed)
            except json.JSONDecodeError:
                pass
            for s in (json_str_fixed, json_str):
                result = _fix_unescaped_quotes(s)
                if result is not None:
                    return result
    except (IndexError, Exception):
        pass

    try:
        start = content.find("[")
        end = content.rfind("]") + 1
        if start != -1 and end > start:
            json_str = content[start:end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            json_str_clean = re.sub(r',\s*([}\]])', r'\1', json_str)
            return json.loads(json_str_clean)
        except (json.JSONDecodeError, UnboundLocalError):
            pass

    try:
        start = content.find("[")
        end = content.rfind("]") + 1
        if start != -1 and end > start:
            json_str = content[start:end]
            json_str_fixed = _fix_missing_opening_quotes(json_str)
            return json.loads(json_str_fixed)
    except (json.JSONDecodeError, Exception):
        pass

    try:
        start = content.find("[")
        end = content.rfind("]") + 1
        if start != -1 and end > start:
            json_str = content[start:end]
            result = _fix_unescaped_quotes(json_str)
            if result is not None:
                return result
    except Exception:
        pass

    try:
        start = content.find("[")
        end = content.rfind("]") + 1
        if start != -1 and end > start:
            json_str = content[start:end]
            json_str_fixed = _fix_missing_opening_quotes(json_str)
            result = _fix_unescaped_quotes(json_str_fixed)
            if result is not None:
                return result
    except Exception:
        pass

    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            json_str = content[start:end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            json_str_clean = re.sub(r',\s*([}\]])', r'\1', json_str)
            return json.loads(json_str_clean)
        except (json.JSONDecodeError, UnboundLocalError):
            pass

    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            json_str = content[start:end]
            json_str_fixed = _fix_missing_opening_quotes(json_str)
            return json.loads(json_str_fixed)
    except (json.JSONDecodeError, Exception):
        pass

    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            json_str = content[start:end]
            result = _fix_unescaped_quotes(json_str)
            if result is not None:
                return result
    except Exception:
        pass

    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            json_str = content[start:end]
            json_str_fixed = _fix_missing_opening_quotes(json_str)
            result = _fix_unescaped_quotes(json_str_fixed)
            if result is not None:
                return result
    except Exception:
        pass

    return None


# ============== 图片编码工具 ==============
def encode_image_to_base64(image_path: Path) -> Optional[str]:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"    [WARN] 读取图片失败 {image_path}: {e}", file=sys.stderr, flush=True)
        return None


def get_mime_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_map.get(suffix, "image/png")


def _open_image_as_rgb(image_path: Path) -> Image.Image:
    """
    用 PIL 打开图片并统一转为 RGB 模式。
    处理动图（只取第一帧）、RGBA/LA/PA/P 等带透明通道的模式。
    """
    img = Image.open(image_path)

    if hasattr(img, "n_frames") and img.n_frames > 1:
        img.seek(0)

    if img.mode in ("RGBA", "LA", "PA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode == "P":
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    return img


def _save_jpeg_to_bytes(img: Image.Image, quality: int) -> bytes:
    """将 PIL Image 以指定 quality 保存为 JPEG 字节。"""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def load_and_compress_image(
    image_path: Path,
    max_size_bytes: int = MAX_IMAGE_SIZE_BYTES,
) -> Optional[Tuple[str, str]]:
    """
    加载图片，统一转为 JPEG 格式，再检测大小是否超过 max_size_bytes。

    Returns:
        (base64_encoded_str, "image/jpeg")  成功时
        None                                失败时
    """
    try:
        original_size_mb = image_path.stat().st_size / (1024 * 1024)

        img = _open_image_as_rgb(image_path)

        jpeg_data = _save_jpeg_to_bytes(img, quality=INITIAL_JPEG_QUALITY)

        if len(jpeg_data) <= max_size_bytes:
            jpeg_mb = len(jpeg_data) / (1024 * 1024)
            if image_path.suffix.lower() not in (".jpg", ".jpeg") or original_size_mb > jpeg_mb * 1.05:
                print(
                    f"    [JPEG] {image_path.name} 转为 JPEG: "
                    f"{original_size_mb:.2f} MB → {jpeg_mb:.2f} MB (quality={INITIAL_JPEG_QUALITY})",
                    file=sys.stderr,
                    flush=True,
                )
            base64_str = base64.b64encode(jpeg_data).decode("utf-8")
            return base64_str, "image/jpeg"

        print(
            f"    [COMPRESS] {image_path.name} 转为 JPEG 后 {len(jpeg_data) / (1024*1024):.2f} MB，"
            f"仍超过限制 {max_size_bytes / (1024*1024):.0f} MB，继续压缩...",
            file=sys.stderr,
            flush=True,
        )

        quality_levels = [90, 85, 80, 70, 60, 50, 40, 30, 20, 10]
        for quality in quality_levels:
            jpeg_data = _save_jpeg_to_bytes(img, quality=quality)
            if len(jpeg_data) <= max_size_bytes:
                compressed_mb = len(jpeg_data) / (1024 * 1024)
                print(
                    f"    [COMPRESS] {image_path.name} 压缩完成: "
                    f"{original_size_mb:.2f} MB → {compressed_mb:.2f} MB (quality={quality})",
                    file=sys.stderr,
                    flush=True,
                )
                base64_str = base64.b64encode(jpeg_data).decode("utf-8")
                return base64_str, "image/jpeg"

        for resize_round in range(1, 11):
            scale = 0.75 ** resize_round
            new_w = max(1, int(img.width * scale))
            new_h = max(1, int(img.height * scale))
            current_img = img.resize((new_w, new_h), Image.LANCZOS)

            for quality in [60, 40, 20, 10]:
                jpeg_data = _save_jpeg_to_bytes(current_img, quality=quality)
                if len(jpeg_data) <= max_size_bytes:
                    compressed_mb = len(jpeg_data) / (1024 * 1024)
                    print(
                        f"    [COMPRESS] {image_path.name} 压缩+缩放完成: "
                        f"{original_size_mb:.2f} MB → {compressed_mb:.2f} MB "
                        f"(quality={quality}, size={new_w}x{new_h})",
                        file=sys.stderr,
                        flush=True,
                    )
                    base64_str = base64.b64encode(jpeg_data).decode("utf-8")
                    return base64_str, "image/jpeg"

        print(
            f"    [WARN] {image_path.name} 无法压缩到 {max_size_bytes / (1024*1024):.0f} MB 以内",
            file=sys.stderr,
            flush=True,
        )
        return None

    except Exception as e:
        print(
            f"    [WARN] 图片压缩失败 {image_path}: {e}",
            file=sys.stderr,
            flush=True,
        )
        return None


# ============== 构建 Original Prompt (Question) 的 Multimodal Parts ==============
def build_original_prompt_parts(
    question_text: str,
    question_img_paths: List[str],
) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """
    构建 original prompt (question) 的 multimodal parts。

    question_text 中可能包含 [IMAGE1], [IMAGE2] 等占位符，
    需要替换为实际图片的 inline_data。如果没有图片，直接返回纯文本 part。

    Args:
        question_text: 原始 prompt 文本（可能含 [IMAGE1] 等占位符）
        question_img_paths: 图片绝对路径字符串列表，可能为空

    Returns:
        (parts_list, error_msg): 成功时 error_msg 为 None，失败时 parts_list 为 None
    """
    if not question_img_paths:
        return [{"text": question_text}], None

    pattern = r'\[IMAGE(\d+)\]'
    parts = []
    last_end = 0

    for match in re.finditer(pattern, question_text):
        # 添加占位符前的文本
        if match.start() > last_end:
            text_segment = question_text[last_end:match.start()]
            if text_segment:
                parts.append({"text": text_segment})

        # 加载对应图片
        img_idx = int(match.group(1)) - 1  # 1-based → 0-based
        if 0 <= img_idx < len(question_img_paths):
            img_path = Path(question_img_paths[img_idx])
            if not img_path.exists():
                return None, f"输入图片缺失: {img_path}"
            result = load_and_compress_image(img_path)
            if result is None:
                return None, f"输入图片加载/压缩失败: {img_path}"
            base64_str, mime_type = result
            parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": base64_str,
                }
            })
        else:
            return None, f"图片索引越界: [IMAGE{img_idx + 1}], 仅有 {len(question_img_paths)} 张图片"

        last_end = match.end()

    # 添加最后一个占位符之后的文本
    if last_end < len(question_text):
        remaining = question_text[last_end:]
        if remaining:
            parts.append({"text": remaining})

    # 如果没有匹配到任何占位符（但 question_img_paths 非空），直接返回整个文本
    if not parts:
        parts.append({"text": question_text})

    return parts, None


# ============== 解析 Prompt 模板为分段 ==============
def parse_template_segments(prompt_template: str) -> Tuple[str, str, str, str]:
    """
    将 prompt 模板按三个占位符分割为四段纯文本。

    模板中占位符的顺序为:
        {original_question} → {model_response} → {checklist}

    Returns:
        (before_question,
         between_question_and_model_response,
         between_model_response_and_checklist,
         after_checklist)
    """
    placeholder_q = "{original_question}"
    placeholder_mr = "{model_response}"
    placeholder_cl = "{checklist}"

    idx_q = prompt_template.index(placeholder_q)
    before_q = prompt_template[:idx_q]
    remaining = prompt_template[idx_q + len(placeholder_q):]

    idx_mr = remaining.index(placeholder_mr)
    between_q_mr = remaining[:idx_mr]
    remaining2 = remaining[idx_mr + len(placeholder_mr):]

    idx_cl = remaining2.index(placeholder_cl)
    between_mr_cl = remaining2[:idx_cl]
    after_cl = remaining2[idx_cl + len(placeholder_cl):]

    return before_q, between_q_mr, between_mr_cl, after_cl


# ============== API 调用函数 ==============
def call_gemini_api(
    parts: List[Dict],
    api_key: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Dict[str, Any]:
    """
    调用 Gemini API，发送预构建的 multimodal parts。

    参数:
        parts: 已构建好的 multimodal parts 列表（包含 text 和 inline_data）
        api_key: API 密钥
        max_retries: 最大重试次数
        timeout: 超时时间（秒）
        temperature: 生成温度
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": _JUDGE_MODEL_NAME,
        "stream": True,
        "dashscope_extend_params": {
            "using_native_protocol": True
        },
        "contents": [
            {
                "parts": parts,
                "role": "user",
            }
        ],
        "generationConfig": {
            "temperature": temperature,
        },
    }

    last_error = None

    for attempt in range(max_retries):
        resp = None
        try:
            text_buffer = ""

            resp = requests.post(
                API_URL, headers=headers, json=payload,
                stream=True, timeout=(30, timeout),
            )

            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                resp.close()

                if resp.status_code in RETRYABLE_STATUS_CODES or resp.status_code >= 500:
                    if attempt < max_retries - 1:
                        if resp.status_code == 429:
                            wait_time = min(60, (2 ** attempt) * 5 + random.uniform(1, 5))
                        else:
                            wait_time = min(30, (attempt + 1) * 3 + random.uniform(1, 3))
                        print(f"    [Gemini] HTTP {resp.status_code}, attempt {attempt+1}/{max_retries}, waiting {wait_time:.1f}s...", file=sys.stderr, flush=True)
                        time.sleep(wait_time)
                        continue
                continue

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue

                chunk_json = line[6:]
                if chunk_json == "[DONE]":
                    break

                try:
                    chunk = json.loads(chunk_json)
                except json.JSONDecodeError:
                    continue

                candidate = chunk.get("candidates", [{}])[0]
                chunk_parts = candidate.get("content", {}).get("parts", [])

                for part in chunk_parts:
                    if "text" in part:
                        text_buffer += part["text"]

            resp.close()

            content = text_buffer.strip()
            if content:
                parsed = parse_json_response(content)
                return {"success": True, "content": content, "parsed": parsed, "error": None}
            else:
                last_error = "返回内容为空"
                if attempt < max_retries - 1:
                    print(f"    [Gemini] Empty response, attempt {attempt+1}/{max_retries}, retrying...", file=sys.stderr, flush=True)
                    time.sleep(random.uniform(2, 5))

        except requests.Timeout:
            last_error = f"请求超时 ({timeout}s)"
            if resp:
                resp.close()
            if attempt < max_retries - 1:
                print(f"    [Gemini] Timeout, attempt {attempt+1}/{max_retries}, retrying...", file=sys.stderr, flush=True)
                time.sleep(random.uniform(3, 6))
        except requests.exceptions.ConnectionError as e:
            last_error = f"连接错误: {str(e)[:100]}"
            if resp:
                resp.close()
            if attempt < max_retries - 1:
                print(f"    [Gemini] Connection error, attempt {attempt+1}/{max_retries}, retrying...", file=sys.stderr, flush=True)
                time.sleep(random.uniform(3, 6))
        except Exception as e:
            last_error = str(e)
            if resp:
                resp.close()
            if attempt < max_retries - 1:
                print(f"    [Gemini] Error: {str(e)[:50]}, attempt {attempt+1}/{max_retries}, retrying...", file=sys.stderr, flush=True)
                time.sleep(random.uniform(2, 5))

    return {"success": False, "content": "", "parsed": None, "error": last_error}


def _gemini_parts_to_openai_content(parts: List[Dict]) -> List[Dict]:
    """Convert Gemini native parts to OpenAI-compatible content blocks."""
    content = []
    for part in parts:
        if "text" in part:
            content.append({"type": "text", "text": part["text"]})
        elif "inline_data" in part:
            d = part["inline_data"]
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{d['mime_type']};base64,{d['data']}"},
            })
    return content


def _parse_openai_stream(resp) -> str:
    """Parse OpenAI-compatible streaming response, skip reasoning_content."""
    text_buffer = ""
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        chunk_json = line[6:]
        if chunk_json == "[DONE]":
            break
        try:
            chunk = json.loads(chunk_json)
        except json.JSONDecodeError:
            continue
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            if "reasoning_content" in delta:
                continue
            c = delta.get("content")
            if c and isinstance(c, str):
                text_buffer += c
    return text_buffer.strip()


def call_gpt_api(
    parts: List[Dict],
    api_key: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Dict[str, Any]:
    """Call GPT model via DashScope OpenAI-compatible endpoint."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    content = _gemini_parts_to_openai_content(parts)
    payload = {
        "model": _JUDGE_MODEL_NAME,
        "stream": True,
        "temperature": temperature,
        "messages": [{"role": "user", "content": content}],
    }

    last_error = None
    for attempt in range(max_retries):
        resp = None
        try:
            resp = requests.post(
                API_URL, headers=headers, json=payload,
                stream=True, timeout=(30, timeout),
            )
            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                resp.close()
                if resp.status_code in RETRYABLE_STATUS_CODES or resp.status_code >= 500:
                    if attempt < max_retries - 1:
                        wait_time = min(60, (2 ** attempt) * 5 + random.uniform(1, 5)) if resp.status_code == 429 else min(30, (attempt + 1) * 3 + random.uniform(1, 3))
                        print(f"    [GPT] HTTP {resp.status_code}, attempt {attempt+1}/{max_retries}, waiting {wait_time:.1f}s...", file=sys.stderr, flush=True)
                        time.sleep(wait_time)
                        continue
                continue

            text = _parse_openai_stream(resp)
            resp.close()
            if text:
                parsed = parse_json_response(text)
                return {"success": True, "content": text, "parsed": parsed, "error": None}
            else:
                last_error = "返回内容为空"
                if attempt < max_retries - 1:
                    print(f"    [GPT] Empty response, attempt {attempt+1}/{max_retries}, retrying...", file=sys.stderr, flush=True)
                    time.sleep(random.uniform(2, 5))
        except requests.Timeout:
            last_error = f"请求超时 ({timeout}s)"
            if resp:
                resp.close()
            if attempt < max_retries - 1:
                print(f"    [GPT] Timeout, attempt {attempt+1}/{max_retries}, retrying...", file=sys.stderr, flush=True)
                time.sleep(random.uniform(3, 6))
        except requests.exceptions.ConnectionError as e:
            last_error = f"连接错误: {str(e)[:100]}"
            if resp:
                resp.close()
            if attempt < max_retries - 1:
                print(f"    [GPT] Connection error, attempt {attempt+1}/{max_retries}, retrying...", file=sys.stderr, flush=True)
                time.sleep(random.uniform(3, 6))
        except Exception as e:
            last_error = str(e)
            if resp:
                resp.close()
            if attempt < max_retries - 1:
                print(f"    [GPT] Error: {str(e)[:50]}, attempt {attempt+1}/{max_retries}, retrying...", file=sys.stderr, flush=True)
                time.sleep(random.uniform(2, 5))

    return {"success": False, "content": "", "parsed": None, "error": last_error}


def call_qwen_api(
    parts: List[Dict],
    api_key: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Dict[str, Any]:
    """Call Qwen model via DashScope OpenAI-compatible endpoint."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    content = _gemini_parts_to_openai_content(parts)
    payload = {
        "model": _JUDGE_MODEL_NAME,
        "stream": True,
        "temperature": temperature,
        "messages": [{"role": "user", "content": content}],
    }

    last_error = None
    for attempt in range(max_retries):
        resp = None
        try:
            resp = requests.post(
                API_URL, headers=headers, json=payload,
                stream=True, timeout=(30, timeout),
            )
            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                resp.close()
                if resp.status_code in RETRYABLE_STATUS_CODES or resp.status_code >= 500:
                    if attempt < max_retries - 1:
                        wait_time = min(60, (2 ** attempt) * 5 + random.uniform(1, 5)) if resp.status_code == 429 else min(30, (attempt + 1) * 3 + random.uniform(1, 3))
                        print(f"    [Qwen] HTTP {resp.status_code}, attempt {attempt+1}/{max_retries}, waiting {wait_time:.1f}s...", file=sys.stderr, flush=True)
                        time.sleep(wait_time)
                        continue
                continue

            text = _parse_openai_stream(resp)
            resp.close()
            if text:
                parsed = parse_json_response(text)
                return {"success": True, "content": text, "parsed": parsed, "error": None}
            else:
                last_error = "返回内容为空"
                if attempt < max_retries - 1:
                    print(f"    [Qwen] Empty response, attempt {attempt+1}/{max_retries}, retrying...", file=sys.stderr, flush=True)
                    time.sleep(random.uniform(2, 5))
        except requests.Timeout:
            last_error = f"请求超时 ({timeout}s)"
            if resp:
                resp.close()
            if attempt < max_retries - 1:
                print(f"    [Qwen] Timeout, attempt {attempt+1}/{max_retries}, retrying...", file=sys.stderr, flush=True)
                time.sleep(random.uniform(3, 6))
        except requests.exceptions.ConnectionError as e:
            last_error = f"连接错误: {str(e)[:100]}"
            if resp:
                resp.close()
            if attempt < max_retries - 1:
                print(f"    [Qwen] Connection error, attempt {attempt+1}/{max_retries}, retrying...", file=sys.stderr, flush=True)
                time.sleep(random.uniform(3, 6))
        except Exception as e:
            last_error = str(e)
            if resp:
                resp.close()
            if attempt < max_retries - 1:
                print(f"    [Qwen] Error: {str(e)[:50]}, attempt {attempt+1}/{max_retries}, retrying...", file=sys.stderr, flush=True)
                time.sleep(random.uniform(2, 5))

    return {"success": False, "content": "", "parsed": None, "error": last_error}


def call_judge_api(
    parts: List[Dict],
    api_key: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Dict[str, Any]:
    """Dispatch to the appropriate judge model API based on _JUDGE_MODEL_TYPE."""
    if _JUDGE_MODEL_TYPE == "gemini":
        return call_gemini_api(parts, api_key, max_retries, timeout, temperature)
    elif _JUDGE_MODEL_TYPE == "gpt":
        return call_gpt_api(parts, api_key, max_retries, timeout, temperature)
    elif _JUDGE_MODEL_TYPE == "qwen":
        return call_qwen_api(parts, api_key, max_retries, timeout, temperature)
    else:
        return {"success": False, "content": "", "parsed": None, "error": f"Unknown judge model type: {_JUDGE_MODEL_TYPE}"}


# ============== 文件锁写入 failed.jsonl ==============
def append_failed(failed_path: Path, data_id: str, error: str):
    record = json.dumps({"id": data_id, "error": error, "timestamp": datetime.now().isoformat()}, ensure_ascii=False)
    try:
        with open(failed_path, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(record + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        print(f"    [WARN] 写入 failed.jsonl 失败: {e}", file=sys.stderr, flush=True)


# ============== 合并 API 结果与 checklist ==============
def merge_result_with_checklist(checklist_data: List[Dict], api_answers: list) -> List[Dict]:
    """
    将 API 返回的 answer 合并到原始 checklist 数据中。
    按 id 匹配，保留 checklist 的所有原始字段，添加 answer 字段。
    """
    answer_map = {}
    for item in api_answers:
        if isinstance(item, dict) and "id" in item:
            answer_map[item["id"]] = item.get("answer", "N/A")

    merged = []
    for item in checklist_data:
        new_item = dict(item)
        new_item["answer"] = answer_map.get(item["id"], "N/A")
        merged.append(new_item)

    return merged


# ============== Worker 初始化 ==============
def _init_worker():
    time.sleep(random.uniform(0.1, 0.5))


# ============== Worker 函数 ==============
def _process_single_item(args: Tuple) -> Dict[str, Any]:
    """
    处理单条多模态理解评测数据。

    流程:
    1. 加载模型输出 ({data_id}.json，提取 response 纯文本)
    2. 加载 checklist
    3. 构建 question 的 multimodal parts（可能含图片占位符替换）
    4. 按模板组装完整 API 请求 parts
       模板顺序: original_question → model_response → checklist
    5. 调用 Gemini API 评分
    6. 合并结果并保存
    """
    (data_id, dimension, model_output_file_str, question_text,
     question_img_paths_json, checklist_path_str, prompt_template,
     eval_output_dir_str, failed_path_str, max_retries, timeout,
     worker_max_retries) = args

    eval_output_dir = Path(eval_output_dir_str)
    failed_path = Path(failed_path_str)
    model_output_file = Path(model_output_file_str)
    question_img_paths = json.loads(question_img_paths_json)
    checklist_path = Path(checklist_path_str)

    output_path = eval_output_dir / dimension / f"{data_id}.json"

    # 断点续传：已处理则跳过
    if output_path.exists():
        return {"id": data_id, "status": "skipped", "error": None}

    # ---- 加载模型输出 JSON 文件，提取 response 纯文本 ----
    if not model_output_file.exists():
        error_msg = f"模型输出文件缺失: {model_output_file}"
        append_failed(failed_path, data_id, error_msg)
        return {"id": data_id, "status": "failed", "error": error_msg}

    try:
        with open(model_output_file, "r", encoding="utf-8") as f:
            response_data = json.load(f)
    except Exception as e:
        error_msg = f"读取模型输出文件失败: {model_output_file}: {e}"
        append_failed(failed_path, data_id, error_msg)
        return {"id": data_id, "status": "failed", "error": error_msg}

    response_text = response_data.get("response", "")
    if not response_text:
        error_msg = f"模型输出 response 为空: {model_output_file}"
        append_failed(failed_path, data_id, error_msg)
        return {"id": data_id, "status": "failed", "error": error_msg}

    # ---- 加载 checklist ----
    try:
        with open(checklist_path, "r", encoding="utf-8") as f:
            checklist_content = f.read().strip()
    except Exception as e:
        error_msg = f"读取 checklist 失败: {checklist_path}: {e}"
        append_failed(failed_path, data_id, error_msg)
        return {"id": data_id, "status": "failed", "error": error_msg}

    try:
        checklist_data = json.loads(checklist_content)
    except json.JSONDecodeError as e:
        error_msg = f"解析 checklist JSON 失败: {checklist_path}: {e}"
        append_failed(failed_path, data_id, error_msg)
        return {"id": data_id, "status": "failed", "error": error_msg}

    # ---- 构建 question (original prompt) parts ----
    original_prompt_parts, err = build_original_prompt_parts(question_text, question_img_paths)
    if err:
        append_failed(failed_path, data_id, err)
        return {"id": data_id, "status": "failed", "error": err}

    # ---- 解析模板分段并组装完整 parts ----
    # 模板占位符顺序: {original_question} → {model_response} → {checklist}
    try:
        before_q, between_q_mr, between_mr_cl, after_cl = parse_template_segments(prompt_template)
    except ValueError as e:
        error_msg = f"Prompt 模板解析失败: {e}"
        append_failed(failed_path, data_id, error_msg)
        return {"id": data_id, "status": "failed", "error": error_msg}

    all_parts = []
    all_parts.append({"text": before_q})
    all_parts.extend(original_prompt_parts)
    # model_response 是纯文本，与 checklist 一起拼接
    all_parts.append({"text": between_q_mr + response_text + between_mr_cl + checklist_content + after_cl})

    # ---- 获取 API key ----
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        error_msg = "未设置 DASHSCOPE_API_KEY 环境变量"
        append_failed(failed_path, data_id, error_msg)
        return {"id": data_id, "status": "failed", "error": error_msg}

    # ---- Worker 重试循环 ----
    last_error = None
    for worker_attempt in range(worker_max_retries):
        try:
            if output_path.exists():
                return {"id": data_id, "status": "skipped", "error": None}

            result = call_judge_api(
                parts=all_parts,
                api_key=api_key,
                max_retries=max_retries,
                timeout=timeout,
            )

            if result["success"]:
                parsed = result["parsed"]
                if parsed is not None and isinstance(parsed, list):
                    merged = merge_result_with_checklist(checklist_data, parsed)

                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path = output_path.with_suffix(".json.tmp")
                    try:
                        with open(tmp_path, "w", encoding="utf-8") as f:
                            json.dump(merged, f, ensure_ascii=False, indent=2)
                        tmp_path.rename(output_path)
                    except Exception as e:
                        if tmp_path.exists():
                            try:
                                tmp_path.unlink()
                            except Exception:
                                pass
                        raise e
                    return {"id": data_id, "status": "success", "error": None}
                else:
                    last_error = f"JSON解析失败，原始内容: {result['content'][:200]}"
                    if worker_attempt < worker_max_retries - 1:
                        wait_time = random.uniform(2, 5) * (worker_attempt + 1)
                        print(f"    [Worker] {data_id}: JSON解析失败, 重试 {worker_attempt+1}/{worker_max_retries}, 等待 {wait_time:.1f}s", file=sys.stderr, flush=True)
                        time.sleep(wait_time)
                        continue
            else:
                last_error = result.get("error", "未知错误")
                if worker_attempt < worker_max_retries - 1:
                    wait_time = random.uniform(2, 5) * (worker_attempt + 1)
                    print(f"    [Worker] {data_id}: API失败 '{last_error[:50]}...', 重试 {worker_attempt+1}/{worker_max_retries}, 等待 {wait_time:.1f}s", file=sys.stderr, flush=True)
                    time.sleep(wait_time)
                    continue

        except Exception as e:
            last_error = str(e)
            if worker_attempt < worker_max_retries - 1:
                wait_time = random.uniform(2, 5) * (worker_attempt + 1)
                print(f"    [Worker] {data_id}: 异常 '{str(e)[:50]}...', 重试 {worker_attempt+1}/{worker_max_retries}, 等待 {wait_time:.1f}s", file=sys.stderr, flush=True)
                time.sleep(wait_time)
                continue

    append_failed(failed_path, data_id, last_error or "未知错误")
    return {"id": data_id, "status": "failed", "error": last_error}


# ============== 数据加载 ==============
def load_prompt_map() -> Dict[str, Tuple[str, List[str]]]:
    """
    从 DATA_DIR 下的所有 .jsonl 和 .json 文件中加载 {id: (question, question_img)} 映射表。

    数据文件包括 charxiv.jsonl, countbench.jsonl, hallusionbench.jsonl, mmbench.jsonl, mmmu.json 等。
    所有赛道的输入都可能是纯文本或文本和图片的交错输入。
    question_img 是参考图片的相对路径列表（基于 DATA_DIR）。
    """
    prompt_map = {}

    if not DATA_DIR.exists():
        print(f"[ERROR] 数据目录不存在: {DATA_DIR}", file=sys.stderr, flush=True)
        sys.exit(1)

    # 加载 .jsonl 文件（逐行 JSON）
    for jsonl_file in sorted(DATA_DIR.glob("*.jsonl")):
        count = 0
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    data_id = item["id"]
                    prompt_text = item["question"]
                    question_img = item.get("question_img", [])
                    prompt_map[data_id] = (prompt_text, question_img)
                    count += 1
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"  [WARN] 解析失败 {jsonl_file.name} 第{line_num}行: {e}", file=sys.stderr, flush=True)
        print(f"  已加载 prompt 映射: {jsonl_file.name} ({count} 条)")

    # 加载 .json 文件（JSON 数组或单个对象）
    for json_file in sorted(DATA_DIR.glob("*.json")):
        count = 0
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for item_idx, item in enumerate(data):
                    try:
                        data_id = item["id"]
                        prompt_text = item["question"]
                        question_img = item.get("question_img", [])
                        prompt_map[data_id] = (prompt_text, question_img)
                        count += 1
                    except KeyError as e:
                        print(f"  [WARN] 解析失败 {json_file.name} 第{item_idx}个元素: 缺少字段 {e}", file=sys.stderr, flush=True)
            elif isinstance(data, dict):
                # 单个对象
                try:
                    data_id = data["id"]
                    prompt_text = data["question"]
                    question_img = data.get("question_img", [])
                    prompt_map[data_id] = (prompt_text, question_img)
                    count = 1
                except KeyError as e:
                    print(f"  [WARN] 解析失败 {json_file.name}: 缺少字段 {e}", file=sys.stderr, flush=True)
            else:
                print(f"  [WARN] {json_file.name} 格式不支持（非数组或对象）", file=sys.stderr, flush=True)
        except (json.JSONDecodeError, Exception) as e:
            print(f"  [WARN] 读取/解析 {json_file.name} 失败: {e}", file=sys.stderr, flush=True)
        print(f"  已加载 prompt 映射: {json_file.name} ({count} 条)")

    print(f"  共加载 {len(prompt_map)} 条 prompt 映射")
    return prompt_map


def load_all_items(
    eval_model_name: str,
    prompt_map: Dict[str, Tuple[str, List[str]]],
    failed_path: Path,
    dimensions: List[str],
) -> List[Tuple[str, str, Path, List[Path], Path, str]]:
    """
    扫描 checklist 目录，构建所有待评测数据列表。

    模型输出为单个 JSON 文件: model_outputs/understanding/{model}/{data_id}.json
        文件内容形如 {"id": "xxx", "response": "...", "error": null}

    输入参考图片路径从 prompt_map 的 question_img 中获取，
    基于 DATA_DIR 拼接绝对路径。

    返回: [(data_id, dimension, model_output_file, question_img_abs_paths, checklist_path, prompt_text), ...]
    """
    all_items = []

    for dimension in dimensions:
        checklist_dim_dir = CHECKLIST_DIR / dimension
        if not checklist_dim_dir.exists():
            print(f"[WARN] Checklist 维度目录不存在: {checklist_dim_dir}", file=sys.stderr, flush=True)
            continue

        count = 0
        missing_output_count = 0
        missing_input_image_count = 0
        missing_prompt_count = 0

        for checklist_file in sorted(checklist_dim_dir.glob("*.json")):
            data_id = checklist_file.stem

            # 查找模型输出文件
            model_output_file = MODEL_OUTPUT_DIR / eval_model_name / f"{data_id}.json"

            if not model_output_file.exists():
                error_msg = f"模型输出文件缺失: {model_output_file}"
                print(f"  [WARN] {error_msg}", file=sys.stderr, flush=True)
                append_failed(failed_path, data_id, error_msg)
                missing_output_count += 1
                continue

            # 获取 prompt
            prompt_entry = prompt_map.get(data_id)
            if prompt_entry is None:
                print(f"  [WARN] 缺少 prompt: {data_id}", file=sys.stderr, flush=True)
                missing_prompt_count += 1
                continue

            prompt_text, question_img_list = prompt_entry

            # 解析输入参考图片绝对路径
            question_img_abs_paths = []
            if question_img_list:
                all_exist = True
                for ref_rel in question_img_list:
                    ref_path = DATA_DIR / ref_rel
                    if not ref_path.exists():
                        error_msg = f"输入图片缺失: {ref_path}"
                        print(f"  [WARN] {error_msg}", file=sys.stderr, flush=True)
                        append_failed(failed_path, data_id, error_msg)
                        missing_input_image_count += 1
                        all_exist = False
                        break
                    question_img_abs_paths.append(ref_path)

                if not all_exist:
                    continue

            all_items.append((
                data_id, dimension, model_output_file,
                question_img_abs_paths, checklist_file, prompt_text
            ))
            count += 1

        if missing_output_count > 0:
            print(f"  [WARN] {dimension}: {missing_output_count} 条数据缺少模型输出（已记录到 failed.jsonl）", file=sys.stderr, flush=True)
        if missing_input_image_count > 0:
            print(f"  [WARN] {dimension}: {missing_input_image_count} 条数据缺少输入图片（已记录到 failed.jsonl）", file=sys.stderr, flush=True)
        if missing_prompt_count > 0:
            print(f"  [WARN] {dimension}: {missing_prompt_count} 条数据缺少 prompt", file=sys.stderr, flush=True)
        print(f"  已加载 {dimension}: {count} 条数据")

    return all_items


def load_prompt_template() -> str:
    """
    加载 eval_prompt.txt 模板，检查必需的占位符。

    模板中应包含以下三个占位符（按此顺序）:
        {original_question}
        {model_response}
        {checklist}
    """
    if not PROMPT_FILE.exists():
        print(f"[ERROR] Prompt 模板不存在: {PROMPT_FILE}", file=sys.stderr, flush=True)
        sys.exit(1)

    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        template = f.read()

    required_placeholders = [
        "{original_question}",
        "{model_response}",
        "{checklist}",
    ]
    for placeholder in required_placeholders:
        if placeholder not in template:
            print(f"[ERROR] Prompt 模板中缺少 {placeholder} 占位符", file=sys.stderr, flush=True)
            sys.exit(1)

    return template


# ============== 主流程 ==============
def run(
    eval_model_name: str,
    num_processes: int = DEFAULT_NUM_PROCESSES,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    limit: Optional[int] = None,
    judge_model: Optional[str] = None,
):
    global _JUDGE_MODEL_NAME
    if judge_model:
        _JUDGE_MODEL_NAME = judge_model
    if not _JUDGE_MODEL_NAME:
        print("[ERROR] 未设置裁判模型；请设置 JUDGE_MODEL 环境变量或用 --judge-model 指定", file=sys.stderr)
        sys.exit(1)
    if not API_URL:
        print("[ERROR] 未设置 JUDGE_API_URL 环境变量（裁判 API 端点）", file=sys.stderr)
        sys.exit(1)

    start_time = time.time()

    print("\n" + "=" * 60)
    print("Multimodal Understanding Checklist 评分")
    print("=" * 60)
    print(f"评测模型: {eval_model_name}")
    print(f"评分模型: {_JUDGE_MODEL_NAME}")
    print(f"进程数: {num_processes}")
    print(f"API 重试次数: {max_retries}")
    print(f"Worker 重试次数: {WORKER_MAX_RETRIES}")
    print(f"超时时间: {timeout}s")
    print(f"图片大小限制: {MAX_IMAGE_SIZE_BYTES / (1024*1024):.0f} MB")
    print(f"图片统一格式: JPEG (初始 quality={INITIAL_JPEG_QUALITY})")
    if limit:
        print(f"限制数量: {limit}")

    model_output_dir = MODEL_OUTPUT_DIR / eval_model_name
    if not model_output_dir.exists():
        print(f"\n[ERROR] 模型输出目录不存在: {model_output_dir}", file=sys.stderr, flush=True)
        print(f"可用模型目录:", file=sys.stderr, flush=True)
        if MODEL_OUTPUT_DIR.exists():
            for d in sorted(MODEL_OUTPUT_DIR.iterdir()):
                if d.is_dir():
                    print(f"  - {d.name}", file=sys.stderr, flush=True)
        sys.exit(1)

    # 自动检测维度
    dimensions = detect_dimensions(CHECKLIST_DIR)
    print(f"检测到维度: {dimensions}")

    output_dir = RESULTS_DIR / eval_model_name
    failed_path = output_dir / "failed.jsonl"

    for dimension in dimensions:
        (output_dir / dimension).mkdir(parents=True, exist_ok=True)

    print("\n加载 Prompt 模板...")
    prompt_template = load_prompt_template()
    print(f"已加载 Prompt 模板: {PROMPT_FILE}")

    print("\n加载 Prompt 映射表...")
    prompt_map = load_prompt_map()

    print("\n加载数据...")
    all_items = load_all_items(eval_model_name, prompt_map, failed_path, dimensions)
    total_loaded = len(all_items)
    print(f"共加载 {total_loaded} 条数据")

    if not all_items:
        print("没有数据需要处理")
        return

    if limit:
        all_items = all_items[:limit]
        print(f"限制处理数量: {len(all_items)}")

    seed = int(time.time() * 1000) + os.getpid()
    random.seed(seed)
    random.shuffle(all_items)
    print(f"\n已打乱数据顺序 (随机种子: {seed})")
    print(f"打乱后前5个ID: {[item[0] for item in all_items[:5]]}")

    worker_args = []
    for data_id, dimension, model_output_file, question_img_abs_paths, checklist_path, prompt_text in all_items:
        question_img_paths_json = json.dumps([str(p) for p in question_img_abs_paths])
        worker_args.append((
            data_id,
            dimension,
            str(model_output_file),
            prompt_text,
            question_img_paths_json,
            str(checklist_path),
            prompt_template,
            str(output_dir),
            str(failed_path),
            max_retries,
            timeout,
            WORKER_MAX_RETRIES,
        ))

    actual_processes = min(num_processes, len(worker_args))
    print(f"\n开始处理，实际进程数: {actual_processes}")
    print("-" * 60)

    success_count = 0
    failed_count = 0
    skipped_count = 0
    processed_count = 0

    with Pool(processes=actual_processes, initializer=_init_worker) as pool:
        for result in pool.imap_unordered(_process_single_item, worker_args):
            processed_count += 1
            status = result["status"]
            data_id = result["id"]

            if status == "success":
                success_count += 1
            elif status == "skipped":
                skipped_count += 1
            elif status == "failed":
                failed_count += 1
                print(f"  [FAILED] {data_id}: {result.get('error', '未知错误')[:80]}", file=sys.stderr, flush=True)

            if processed_count % 10 == 0 or processed_count == len(worker_args):
                elapsed = time.time() - start_time
                speed = processed_count / elapsed if elapsed > 0 else 0
                remaining = (len(worker_args) - processed_count) / speed if speed > 0 else 0
                print(
                    f"  进度: {processed_count}/{len(worker_args)} "
                    f"(成功: {success_count}, 跳过: {skipped_count}, 失败: {failed_count}) "
                    f"速度: {speed:.1f} items/s, 预计剩余: {remaining/60:.1f} min",
                    flush=True,
                )

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("处理完成")
    print("=" * 60)
    print(f"评测模型: {eval_model_name}")
    print(f"总耗时: {elapsed/60:.1f} 分钟")
    print(f"总数据: {len(worker_args)}")
    print(f"成功: {success_count}")
    print(f"跳过(已存在): {skipped_count}")
    print(f"失败: {failed_count}")
    print(f"结果目录: {output_dir}")
    if failed_count > 0:
        print(f"失败记录: {failed_path}")


def main():
    parser = argparse.ArgumentParser(description="Multimodal Understanding Checklist 评分脚本")

    parser.add_argument(
        "--model-name",
        type=str,
        nargs="+",
        required=True,
        help="待评测模型名称（可指定多个，如 model_a model_b）",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=DEFAULT_NUM_PROCESSES,
        help=f"并行进程数 (默认: {DEFAULT_NUM_PROCESSES})",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"API 调用最大重试次数 (默认: {DEFAULT_MAX_RETRIES})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"API 调用超时时间/秒 (默认: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="每个模型最大处理数量（用于测试）",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="裁判模型 id（覆盖 JUDGE_MODEL 环境变量；不指定则读取该环境变量）",
    )

    args = parser.parse_args()

    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("[ERROR] 请设置 DASHSCOPE_API_KEY 环境变量", file=sys.stderr)
        sys.exit(1)

    for model_name in args.model_name:
        run(
            eval_model_name=model_name,
            num_processes=args.num_processes,
            max_retries=args.max_retries,
            timeout=args.timeout,
            limit=args.limit,
            judge_model=args.judge_model,
        )


if __name__ == "__main__":
    main()
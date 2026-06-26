#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Text-to-Image (T2I) checklist (SVP) evaluation.

Sends each model output together with the task prompt and the per-item checklist to a judge model
(default: gemini-3-flash-preview, via the DashScope OpenAI-compatible endpoint) and records a Yes/No
answer for every checklist item. Supports multiprocessing, resume (skips finished items) and retries.
Output mirrors the checklist with an added "answer" field, under results/<model>/<dimension>/<id>.json.
Run score_and_rank.py afterwards to aggregate per-dimension and overall scores.

Usage:
    # one model
    python run_eval.py --model-name nano_banana --num-processes 50
    # several models
    python run_eval.py --model-name nano_banana gpt_image_1 --num-processes 50
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


# ============== 路径配置 ==============
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
CHECKLIST_DIR = PROJECT_DIR / "eval_checklist" / "t2i"
IMAGE_BASE_DIR = PROJECT_DIR / "model_outputs" / "t2i"
DATA_DIR = PROJECT_DIR / "eval_data" / "t2i"
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

# ============== 维度列表 ==============
DIMENSIONS = [
    "composition",
    "structure",
    "style",
    "text_rendering",
    "world_knowledge_and_reasoning",
]

# ============== 默认配置 ==============
DEFAULT_MAX_RETRIES = 6
DEFAULT_TIMEOUT = 180
DEFAULT_NUM_PROCESSES = 100
DEFAULT_TEMPERATURE = 0.1
WORKER_MAX_RETRIES = 3

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 400, 408, 409, 423, 449}


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

    尝试顺序:
    1. 直接解析
    2. 提取 ```json ... ``` 中的内容
    3. 提取 ``` ... ``` 中的内容（去除 trailing comma）
    3.5. 提取未闭合的 ```json 代码块
    4. 查找首个 [ 到最后一个 ]（列表格式）
    5. 查找首个 { 到最后一个 }（字典格式）
    6. 对以上各步提取的内容，若 json.loads 失败，尝试修复未转义引号后重试
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


# ============== API 调用函数 ==============
def call_gemini_api(
    prompt_text: str,
    image_base64: str,
    mime_type: str,
    api_key: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Dict[str, Any]:
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
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_base64,
                        }
                    },
                    {"text": prompt_text},
                ],
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
                parts = candidate.get("content", {}).get("parts", [])

                for part in parts:
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
    prompt_text: str,
    image_base64: str,
    mime_type: str,
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
    content = [
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}},
        {"type": "text", "text": prompt_text},
    ]
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
    prompt_text: str,
    image_base64: str,
    mime_type: str,
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
    content = [
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}},
        {"type": "text", "text": prompt_text},
    ]
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
    prompt_text: str,
    image_base64: str,
    mime_type: str,
    api_key: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Dict[str, Any]:
    """Dispatch to the appropriate judge model API based on _JUDGE_MODEL_TYPE."""
    if _JUDGE_MODEL_TYPE == "gemini":
        return call_gemini_api(prompt_text, image_base64, mime_type, api_key, max_retries, timeout, temperature)
    elif _JUDGE_MODEL_TYPE == "gpt":
        return call_gpt_api(prompt_text, image_base64, mime_type, api_key, max_retries, timeout, temperature)
    elif _JUDGE_MODEL_TYPE == "qwen":
        return call_qwen_api(prompt_text, image_base64, mime_type, api_key, max_retries, timeout, temperature)
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
    (data_id, dimension, image_path_str, checklist_path_str,
     prompt_text, prompt_template, output_dir_str, failed_path_str,
     max_retries, timeout, worker_max_retries) = args

    output_dir = Path(output_dir_str)
    failed_path = Path(failed_path_str)
    image_path = Path(image_path_str)
    checklist_path = Path(checklist_path_str)

    output_path = output_dir / dimension / f"{data_id}.json"

    if output_path.exists():
        return {"id": data_id, "status": "skipped", "error": None}

    # 检查图片是否存在
    if not image_path.exists():
        error_msg = f"图片缺失: {image_path}"
        append_failed(failed_path, data_id, error_msg)
        return {"id": data_id, "status": "failed", "error": error_msg}

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

    image_base64 = encode_image_to_base64(image_path)
    if image_base64 is None:
        error_msg = f"图片编码失败: {image_path}"
        append_failed(failed_path, data_id, error_msg)
        return {"id": data_id, "status": "failed", "error": error_msg}

    mime_type = get_mime_type(image_path)

    full_prompt = prompt_template.replace("{insert_image_here}", "[见图片]")
    full_prompt = full_prompt.replace("{insert_prompt_here}", prompt_text)
    full_prompt = full_prompt.replace("{insert_checklist_here}", checklist_content)

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        error_msg = "未设置 DASHSCOPE_API_KEY 环境变量"
        append_failed(failed_path, data_id, error_msg)
        return {"id": data_id, "status": "failed", "error": error_msg}

    last_error = None
    for worker_attempt in range(worker_max_retries):
        try:
            if output_path.exists():
                return {"id": data_id, "status": "skipped", "error": None}

            result = call_judge_api(
                prompt_text=full_prompt,
                image_base64=image_base64,
                mime_type=mime_type,
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
def load_prompt_map() -> Dict[str, str]:
    """
    从 DATA_DIR 下的所有 .jsonl 文件中加载 {id: question} 映射表
    """
    prompt_map = {}

    if not DATA_DIR.exists():
        print(f"[ERROR] 数据目录不存在: {DATA_DIR}", file=sys.stderr, flush=True)
        sys.exit(1)

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
                    prompt_map[data_id] = prompt_text
                    count += 1
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"  [WARN] 解析失败 {jsonl_file.name} 第{line_num}行: {e}", file=sys.stderr, flush=True)
        print(f"  已加载 prompt 映射: {jsonl_file.name} ({count} 条)")

    print(f"  共加载 {len(prompt_map)} 条 prompt 映射")
    return prompt_map


def load_all_items(eval_model_name: str, prompt_map: Dict[str, str], failed_path: Path) -> List[Tuple[str, str, Path, Path, str]]:
    """
    扫描 checklist 目录，构建所有待评测数据列表。
    图片为扁平存储: model_outputs/t2i/{model}/{data_id}.{ext}
    图片缺失时记录到 failed.jsonl 并跳过。
    """
    all_items = []

    for dimension in DIMENSIONS:
        checklist_dim_dir = CHECKLIST_DIR / dimension
        if not checklist_dim_dir.exists():
            print(f"[WARN] Checklist 维度目录不存在: {checklist_dim_dir}", file=sys.stderr, flush=True)
            continue

        count = 0
        missing_image_count = 0
        missing_prompt_count = 0
        for checklist_file in sorted(checklist_dim_dir.glob("*.json")):
            data_id = checklist_file.stem

            # 扁平目录结构查找图片
            image_dir = IMAGE_BASE_DIR / eval_model_name
            image_path = None
            for ext in [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"]:
                candidate = image_dir / f"{data_id}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path is None:
                error_msg = f"图片缺失: {image_dir / data_id}.*"
                print(f"  [WARN] {error_msg}", file=sys.stderr, flush=True)
                append_failed(failed_path, data_id, error_msg)
                missing_image_count += 1
                continue

            prompt_text = prompt_map.get(data_id)
            if prompt_text is None:
                print(f"  [WARN] 缺少 prompt: {data_id}", file=sys.stderr, flush=True)
                missing_prompt_count += 1
                continue

            all_items.append((data_id, dimension, image_path, checklist_file, prompt_text))
            count += 1

        if missing_image_count > 0:
            print(f"  [WARN] {dimension}: {missing_image_count} 条数据缺少图片（已记录到 failed.jsonl）", file=sys.stderr, flush=True)
        if missing_prompt_count > 0:
            print(f"  [WARN] {dimension}: {missing_prompt_count} 条数据缺少 prompt", file=sys.stderr, flush=True)
        print(f"  已加载 {dimension}: {count} 条数据")

    return all_items


def load_prompt_template() -> str:
    if not PROMPT_FILE.exists():
        print(f"[ERROR] Prompt 模板不存在: {PROMPT_FILE}", file=sys.stderr, flush=True)
        sys.exit(1)

    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        template = f.read()

    required_placeholders = ["{insert_image_here}", "{insert_checklist_here}", "{insert_prompt_here}"]
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
    print("T2I Benchmark Checklist 评分")
    print("=" * 60)
    print(f"评测模型: {eval_model_name}")
    print(f"评分模型: {_JUDGE_MODEL_NAME}")
    print(f"进程数: {num_processes}")
    print(f"API 重试次数: {max_retries}")
    print(f"Worker 重试次数: {WORKER_MAX_RETRIES}")
    print(f"超时时间: {timeout}s")
    if limit:
        print(f"限制数量: {limit}")

    image_model_dir = IMAGE_BASE_DIR / eval_model_name
    if not image_model_dir.exists():
        print(f"\n[ERROR] 模型图片目录不存在: {image_model_dir}", file=sys.stderr, flush=True)
        print(f"可用模型目录:", file=sys.stderr, flush=True)
        if IMAGE_BASE_DIR.exists():
            for d in sorted(IMAGE_BASE_DIR.iterdir()):
                if d.is_dir():
                    print(f"  - {d.name}", file=sys.stderr, flush=True)
        sys.exit(1)

    output_dir = RESULTS_DIR / eval_model_name
    failed_path = output_dir / "failed.jsonl"

    for dimension in DIMENSIONS:
        (output_dir / dimension).mkdir(parents=True, exist_ok=True)

    print("\n加载 Prompt 模板...")
    prompt_template = load_prompt_template()
    print(f"已加载 Prompt 模板: {PROMPT_FILE}")

    print("\n加载 Prompt 映射表...")
    prompt_map = load_prompt_map()

    print("\n加载数据...")
    all_items = load_all_items(eval_model_name, prompt_map, failed_path)
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
    for data_id, dimension, image_path, checklist_path, prompt_text in all_items:
        worker_args.append((
            data_id,
            dimension,
            str(image_path),
            str(checklist_path),
            prompt_text,
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
    parser = argparse.ArgumentParser(description="T2I Benchmark Checklist 评分脚本")

    parser.add_argument(
        "--model-name",
        type=str,
        nargs="+",
        required=True,
        help="待评测模型名称（可指定多个，如 nano_banana_flash2 gpt_image_1）",
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

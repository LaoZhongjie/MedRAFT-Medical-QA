import os
from dotenv import load_dotenv
from dashscope import Generation


def qwen_complete(prompt: str, model: str = None, temperature: float = 0.2) -> str:
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return "[ERROR] DASHSCOPE_API_KEY is not set."

    # 地域切换：优先使用 .env 中的 DASHSCOPE_BASE_URL（北京默认），可选 DASHSCOPE_INTL=1 切新加坡
    try:
        import dashscope
        base_url = os.getenv("DASHSCOPE_BASE_URL")
        if base_url:
            dashscope.base_http_api_url = base_url
        elif os.getenv("DASHSCOPE_INTL") == "1":
            dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
    except Exception:
        pass

    # 默认模型使用可用的标识，允许在 .env 中通过 QWEN_MODEL 覆盖
    model = model or os.getenv("QWEN_MODEL", "qwen3-max")

    def extract_messages(p: str):
        # 从 RAG Prompt 中提取 System/Human 两段，构造 messages
        sys_tag = "System:"
        human_tag = "Human:"
        sys_idx = p.find(sys_tag)
        human_idx = p.find(human_tag)
        if sys_idx != -1 and human_idx != -1 and human_idx > sys_idx:
            sys_content = p[sys_idx + len(sys_tag):human_idx].strip()
            user_content = p[human_idx + len(human_tag):].strip()
            msgs = []
            if sys_content:
                msgs.append({"role": "system", "content": sys_content})
            msgs.append({"role": "user", "content": user_content or p})
            return msgs
        return [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": p}]

    def parse_resp(r):
        if r is None:
            return ""
        # 优先解析 message 结构
        o = getattr(r, "output", None)
        if isinstance(o, dict):
            choices = o.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
                    if isinstance(content, list) and content:
                        first = content[0]
                        if isinstance(first, dict):
                            t = first.get("text")
                            if isinstance(t, str) and t.strip():
                                return t.strip()
        # 兼容 text 结构
        txt = getattr(r, "output_text", None)
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
        return ""

    try:
        # 1) 首选：messages + message 格式
        msgs = extract_messages(prompt)
        resp = Generation.call(
            api_key=api_key,
            model=model,
            messages=msgs,
            result_format="message",
            temperature=temperature,
        )
        out = parse_resp(resp)
        if out:
            return out

        # 2) 回退：prompt/input + text 格式
        resp2 = None
        try:
            resp2 = Generation.call(
                api_key=api_key,
                model=model,
                prompt=prompt,
                result_format="text",
                temperature=temperature,
            )
        except TypeError:
            resp2 = Generation.call(
                api_key=api_key,
                model=model,
                input=prompt,
                result_format="text",
                temperature=temperature,
            )
        out2 = parse_resp(resp2)
        if out2:
            return out2

        # 3) 仍为空：输出更清晰的错误信息
        sc = getattr(resp, "status_code", None) or getattr(resp2, "status_code", None)
        code = getattr(resp, "code", None) or getattr(resp2, "code", None)
        msg = getattr(resp, "message", None) or getattr(resp2, "message", None)
        rid = getattr(resp, "request_id", None) or getattr(resp2, "request_id", None)
        return f"[ERROR] Empty response from Qwen API (status_code={sc}, code={code}, message={msg}, request_id={rid})"
    except Exception as e:
        return f"[ERROR] Qwen API exception: {e}"


def _configure_dashscope_base_url():
    try:
        import dashscope
        base_url = os.getenv("DASHSCOPE_BASE_URL")
        if base_url:
            dashscope.base_http_api_url = base_url
        elif os.getenv("DASHSCOPE_INTL") == "1":
            dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
    except Exception:
        pass

def parse_resp(r):
    if r is None:
        return ""
    # 优先解析 message 结构（使用 dict.get，避免 KeyError）
    try:
        o = r.get("output", None)
    except Exception:
        o = None
    if isinstance(o, dict):
        choices = o.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                if isinstance(content, list) and content:
                    first = content[0]
                    if isinstance(first, dict):
                        t = first.get("text")
                        if isinstance(t, str) and t.strip():
                            return t.strip()
    # 兼容 text 结构（使用 dict.get，避免 KeyError）
    try:
        txt = r.get("output_text", None)
    except Exception:
        txt = None
    if isinstance(txt, str) and txt.strip():
        return txt.strip()
    return ""

def qwen_chat(system: str, user: str, model: str = None, temperature: float = 0.3, max_tokens: int = 800) -> str:
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return "[ERROR] DASHSCOPE_API_KEY is not set."

    _configure_dashscope_base_url()

    model = model or os.getenv("QWEN_MODEL", "qwen3-max")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    try:
        resp = Generation.call(
            api_key=api_key,
            model=model,
            messages=messages,
            result_format="message",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except TypeError:
        resp = Generation.call(
            api_key=api_key,
            model=model,
            messages=messages,
            result_format="message",
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    out = parse_resp(resp)
    if out:
        return out

    # 安全读取响应字段，避免 KeyError
    def _safe_field(r, key):
        try:
            return r.get(key, None)
        except Exception:
            try:
                return r[key]
            except Exception:
                return None

    sc = _safe_field(resp, "status_code")
    code = _safe_field(resp, "code")
    msg = _safe_field(resp, "message")
    rid = _safe_field(resp, "request_id")
    return f"[ERROR] Empty response from Qwen API (status_code={sc}, code={code}, message={msg}, request_id={rid})"
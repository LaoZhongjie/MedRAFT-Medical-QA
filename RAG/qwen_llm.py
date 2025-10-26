import os
from dotenv import load_dotenv
from dashscope import Generation


def qwen_complete(prompt: str, model: str = None, temperature: float = 0.2) -> str:
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return "[ERROR] DASHSCOPE_API_KEY is not set."

    model = model or os.getenv("QWEN_MODEL", "qwen2.5-7b-instruct")
    try:
        # Prefer using 'prompt' per dashscope Generation API; fallback to 'input'
        try:
            resp = Generation.call(
                model=model,
                prompt=prompt,
                temperature=temperature,
                api_key=api_key,
            )
        except TypeError:
            resp = Generation.call(
                model=model,
                input=prompt,
                temperature=temperature,
                api_key=api_key,
            )
        # dashscope新版输出结构
        try:
            return resp.output_text
        except Exception:
            pass
        # 兼容旧结构
        if resp and isinstance(resp, dict) and resp.get("output"):
            choices = resp["output"].get("choices", [])
            if choices:
                return choices[0].get("text") or ""
        return "[ERROR] Empty response from Qwen API"
    except Exception as e:
        return f"[ERROR] Qwen API exception: {e}"
import os
from typing import List, Optional

class AnswerGenerator:
    def __init__(self, openai_api_key: Optional[str] = None):
        # Provider selection: prefer explicit env, otherwise infer from provided key
        self.provider = os.environ.get("LLM_PROVIDER")
        if not self.provider:
            if os.environ.get("DASHSCOPE_API_KEY"):
                self.provider = "qwen"
            elif openai_api_key or os.environ.get("OPENAI_API_KEY"):
                self.provider = "openai"
            else:
                self.provider = "none"

        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")
        self.qwen_model = os.environ.get("QWEN_MODEL", "qwen-turbo")

        # Lazy-init clients to avoid import errors on py38
        self._openai_client = None
        self._dashscope = None

        if self.provider == "openai" and self.openai_api_key:
            try:
                # Prefer OpenAI v1 client if available
                from openai import OpenAI  # type: ignore
                self._openai_client = OpenAI(api_key=self.openai_api_key)
            except Exception:
                # Fallback to legacy 0.x library
                import openai  # type: ignore
                openai.api_key = self.openai_api_key
                self._openai_client = openai
        elif self.provider == "qwen" and self.dashscope_api_key:
            try:
                import dashscope  # type: ignore
                dashscope.api_key = self.dashscope_api_key
                self._dashscope = dashscope
            except Exception:
                self._dashscope = None

    def generate(self, question: str, contexts: List[str]) -> str:
        if self.provider == "openai" and self._openai_client is not None:
            system_prompt = (
                "You are a medical information assistant. Use the provided contexts "
                "to answer accurately and concisely. Cite sources by title or file name."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._format_user_content(question, contexts)},
            ]
            try:
                # v1 style
                if hasattr(self._openai_client, "chat"):
                    resp = self._openai_client.chat.completions.create(
                        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                        messages=messages,
                        temperature=0.2,
                        max_tokens=800,
                    )
                    return resp.choices[0].message.content or ""
                # 0.x style
                else:
                    resp = self._openai_client.ChatCompletion.create(
                        model=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
                        messages=messages,
                        temperature=0.2,
                        max_tokens=800,
                    )
                    return resp["choices"][0]["message"]["content"]
            except Exception as e:
                return f"[LLM error: {e}]\n\n" + self._extractive_answer(question, contexts)

        if self.provider == "qwen" and self._dashscope is not None:
            try:
                content = self._format_user_content(question, contexts)
                resp = self._dashscope.ChatCompletion.create(
                    model=self.qwen_model,
                    messages=[
                        {"role": "system", "content": "You are a bilingual medical RAG assistant."},
                        {"role": "user", "content": content},
                    ],
                    temperature=0.2,
                )
                # Adapt to dashscope response format
                if hasattr(resp, "output") and "text" in resp.output:
                    return resp.output["text"]
                if isinstance(resp, dict):
                    # Some dashscope versions return dict
                    choices = resp.get("choices") or []
                    if choices:
                        msg = choices[0].get("message", {})
                        return msg.get("content", "")
                return ""
            except Exception as e:
                return f"[LLM error: {e}]\n\n" + self._extractive_answer(question, contexts)

        # Offline fallback
        return self._extractive_answer(question, contexts)

    def _format_user_content(self, question: str, contexts: List[str]) -> str:
        header = "Answer the question using the following contexts.\n\n"
        joined = "\n\n---\n\n".join(contexts)
        return f"{header}Question: {question}\n\nContexts:\n{joined}"

    def _extractive_answer(self, question: str, contexts: List[str]) -> str:
        # Simplistic extractive method: return top-2 contexts
        head = "[Offline mode] Returning top contexts as answer.\n\n"
        return head + "\n\n".join(contexts[:2])
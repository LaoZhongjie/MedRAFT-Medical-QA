import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_chain import load_vectorstore, make_rag_chain
from qwen_llm import qwen_complete

app = FastAPI(title="医学RAG问答系统", version="0.1.0")

vs = None


class QueryReq(BaseModel):
    query: str
    topk: int = 5
    lang: str = "zh"  # zh 或 en


class QueryResp(BaseModel):
    prompt: str
    answer: str


@app.on_event("startup")
def on_startup():
    global vs
    load_dotenv()
    persist_dir = os.getenv("CHROMA_DIR", "./chroma_store")
    vs = load_vectorstore(persist_dir)


@app.post("/query", response_model=QueryResp)
def query(req: QueryReq):
    chain = make_rag_chain(vs, query_lang=req.lang)
    prompt = chain.invoke(req.query)
    answer = qwen_complete(prompt)
    return QueryResp(prompt=prompt, answer=answer)


@app.get("/health")
def health():
    return {"status": "ok"}
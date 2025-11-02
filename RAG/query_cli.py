import argparse
import os
from dotenv import load_dotenv
from rich.console import Console

from rag_chain import load_vectorstore, make_rag_chain
from qwen_llm import qwen_complete

console = Console()


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Query RAG CLI")
    parser.add_argument("-q", "--query", type=str, required=True, help="Query text")
    parser.add_argument("-k", "--topk", type=int, default=5, help="Retriever top-k")
    parser.add_argument("-l", "--lang", type=str, default="zh", choices=["zh", "en"], help="Answer language")
    args = parser.parse_args()

    vs = load_vectorstore(os.getenv("CHROMA_DIR", "./chroma_store"))
    chain = make_rag_chain(vs, query_lang=args.lang, top_k=args.topk)

    prompt_str = chain.invoke(args.query)
    # 真正调用千问3 LLM
    output = qwen_complete(prompt_str)

    console.print("[bold]Prompt[/bold]\n" + prompt_str)
    console.print("\n[bold]Answer[/bold]\n" + output)


if __name__ == "__main__":
    main()
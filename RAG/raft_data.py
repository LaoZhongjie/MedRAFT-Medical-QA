from __future__ import annotations
import argparse
import json
import random
import uuid
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from loaders import load_documents
from splitter import split_documents
from embedder import build_embeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ---------------- args ----------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", type=Path, default=Path("teacher_dataset2.jsonl"))
    p.add_argument("--chroma", type=Path, default=Path("chroma_store"))
    p.add_argument("--data_dir", type=Path, default=Path("./data"))
    p.add_argument("--output", type=Path, default=Path("raft_dataset_with_distractors.jsonl"))
    p.add_argument("--num_hard", type=int, default=2)
    p.add_argument("--num_rand", type=int, default=2)
    p.add_argument("--top_k_pool", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_excerpt_chars", type=int, default=800)
    return p.parse_args()

# ---------------- helpers ----------------
def doc_display_str(d: Document, max_chars: int = 800) -> str:
    meta = d.metadata or {}
    cat = meta.get("category", "-")
    dn = meta.get("disease_name", "-")
    sec = meta.get("section", "-")
    fn = meta.get("file_name", "-")
    src = meta.get("source", "-")
    text = (d.page_content or "").strip()
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0] + " ...[truncated]"
    return f"[{cat}] {dn} | {sec} | {fn}\n摘录:\n{text}\nSource: {src}"

def find_oracle_doc(oracle_content: str, all_chunks: List[Document]) -> Optional[Document]:
    oc = oracle_content.strip()
    for d in all_chunks:
        if (d.page_content or "").strip() == oc:
            return d
    short = oc[:200].strip()
    for d in all_chunks:
        if short and short in (d.page_content or ""):
            return d
    return None

def unique_sample_exclude(pool: List[Document], exclude_set: set, k: int) -> List[Document]:
    candidates = [d for d in pool if id(d) not in exclude_set]
    if not candidates:
        return []
    k = min(k, len(candidates))
    return random.sample(candidates, k)

# ---------------- main ----------------
def main():
    args = get_args()
    random.seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # 1. load teacher dataset
    with args.teacher.open("r", encoding="utf-8") as f:
        teacher_lines = [json.loads(ln) for ln in f if ln.strip()]
    print(f"[info] Loaded {len(teacher_lines)} teacher records")

    # 2. build embeddings and open Chroma
    embeddings = build_embeddings()
    vs = Chroma(persist_directory=str(args.chroma), embedding_function=embeddings)
    print(f"[info] Opened Chroma at {args.chroma}")

    # 3. load original documents & split
    load_res = load_documents(str(args.data_dir), exts=[".json"])
    all_orig_docs = load_res.documents
    all_chunks = split_documents(all_orig_docs)
    print(f"[info] Loaded {len(all_orig_docs)} raw docs -> {len(all_chunks)} chunks")

    out_f = args.output.open("w", encoding="utf-8")
    count = 0

    for rec in tqdm(teacher_lines, desc="Processing teacher samples"):
        question = rec.get("question_text")
        teacher_label = rec.get("teacher_soft_label")
        oracle_context = rec.get("oracle_context") or rec.get("context")
        context_docs_meta = rec.get("context_docs") or []

        # --- find oracle doc ---
        oracle_doc = None
        if oracle_context:
            oracle_doc = find_oracle_doc(oracle_context, all_chunks)
        if not oracle_doc and context_docs_meta:
            meta0 = context_docs_meta[0]
            fname = meta0.get("file_name")
            section = meta0.get("section")
            for d in all_chunks:
                if fname and section and d.metadata.get("file_name") == fname and d.metadata.get("section") == section:
                    oracle_doc = d
                    break
        if not oracle_doc and oracle_context:
            hits = vs.similarity_search(oracle_context, k=1)
            if hits:
                oracle_doc = hits[0]
        if not oracle_doc:
            oracle_doc = random.choice(all_chunks) if all_chunks else None

        # --- build distractors ---
        exclude_ids = {id(oracle_doc)} if oracle_doc else set()

        # semantic hard negatives
        hard_negs = []
        if oracle_doc:
            try:
                hits = vs.similarity_search(oracle_doc.page_content, k=args.num_hard + 5)
            except Exception:
                hits = []
            for h in hits:
                if id(h) not in exclude_ids:
                    hard_negs.append(h)
                    exclude_ids.add(id(h))
                if len(hard_negs) >= args.num_hard:
                    break

        # random negatives (prefer different disease_name)
        random_pool = [d for d in all_chunks if id(d) not in exclude_ids]
        oracle_dn = oracle_doc.metadata.get("disease_name") if oracle_doc else None
        diff_pool = [d for d in random_pool if d.metadata.get("disease_name") != oracle_dn]
        if len(diff_pool) >= args.num_rand:
            rand_negs = random.sample(diff_pool, args.num_rand)
        else:
            rand_negs = unique_sample_exclude(random_pool, exclude_ids, args.num_rand)

        distractors = list({id(d): d for d in hard_negs + rand_negs}.values())
        if len(distractors) < (args.num_hard + args.num_rand):
            more = unique_sample_exclude(all_chunks, exclude_ids | {id(x) for x in distractors}, (args.num_hard + args.num_rand) - len(distractors))
            distractors.extend(more)

        # final context
        final_docs = [oracle_doc] if oracle_doc else []
        final_docs.extend(distractors)
        random.shuffle(final_docs)
        context_str = "\n\n".join(doc_display_str(d, max_chars=args.max_excerpt_chars) for d in final_docs)

        out_sample = {
            "id": str(uuid.uuid4()),
            "question": question,
            "context": context_str,
            "oracle_context": oracle_doc.page_content if oracle_doc else oracle_context,
            "cot_answer": teacher_label,
            "disease_name": oracle_doc.metadata.get("disease_name") if oracle_doc else None,
            "category": oracle_doc.metadata.get("category") if oracle_doc else None,
            "lang": oracle_doc.metadata.get("doc_lang") if oracle_doc else rec.get("lang", "zh")
        }

        out_f.write(json.dumps(out_sample, ensure_ascii=False) + "\n")
        count += 1

    out_f.close()
    print(f"[done] wrote {count} samples to {args.output}")
    

if __name__ == "__main__":
    main()



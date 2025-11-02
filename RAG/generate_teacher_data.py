import os
import json
import random
import argparse
from typing import List, Dict, Tuple
from dotenv import load_dotenv

from rag_chain import load_vectorstore, make_teacher_messages
from qwen_llm import qwen_chat

def iter_diseases_from_data_dir(data_dir: str) -> List[Tuple[str, str]]:
    diseases = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if not fn.lower().endswith(".json"):
                continue
            fp = os.path.join(root, fn)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        name = item.get("disease_name")
                        cat = item.get("category")
                        if name:
                            diseases.append((str(name), str(cat) if cat else ""))
            except Exception:
                # 跳过无效/非列表结构的文件
                pass
    # 去重
    uniq = {}
    for name, cat in diseases:
        if name not in uniq:
            uniq[name] = cat
    return [(n, uniq[n]) for n in uniq.keys()]

ZH_TEMPLATES = [
    "关于{d}的典型症状与分期是什么？如何识别？",
    "若患者疑似{d}，初步诊断建议与鉴别诊断有哪些？",
    "{d}的急性期治疗与预防性治疗如何选择？请给出分步建议。",
    "确诊{d}需要做哪些检查？关键阳性/阴性依据是什么？",
    "出现哪些红旗症状提示可能为{d}，需要紧急就医？",
    "在合并其他{c}相关共病时如何管理{d}？给出注意事项。",
    "{d}常见诱因与预防策略有哪些？生活方式如何调整？",
    "针对儿童/老年人患{d}，诊疗策略有哪些差异？",
    "{d}的药物治疗有哪些禁忌或相互作用风险？",
    "{d}的病理生理机制是什么？发病原因有哪些？",
    "如何向患者解释{d}的病情？需要注意哪些沟通要点？",
    "{d}的并发症有哪些？如何预防与处理？",
    "患有{d}的患者在饮食上有什么特殊要求？",
    "{d}的康复治疗方案包括哪些内容？",
    "如何评估{d}患者的病情严重程度？",
    "{d}在不同季节的发病特点和预防措施？",
    "妊娠期患{d}有哪些特殊考虑和处理原则？",
    "{d}的家庭护理要点和注意事项有哪些？",
    "如何制定{d}患者的长期随访计划？",
    "{d}与其他疾病的关联性和共同危险因素？",
    "我最近出现的症状会是{d}吗？需要注意什么？",
    "如果我怀疑自己得了{d}，我应该先做哪些检查？",
    "确诊{d}之后，我日常应该怎么管理和复查？",
    "治疗{d}的药物我该怎么吃？会有哪些副作用？",
    "我正在备孕/妊娠，如果患有{d}，需要怎么处理？",
    "孩子或老人得了{d}，家属照护有哪些要点？",
    "哪些情况属于{d}的紧急信号，我需要马上就医？",
    "我同时还有其他{c}相关疾病，治疗{d}时要注意什么？",
    "我的饮食和运动如何调整，有助于控制{d}？",
    "工作和学习是否会受{d}影响？怎样安排作息更合理？",
    "请用通俗的语言给我解释一下{d}是什么病。",
    "我怎么知道{d}控制得好不好？需要关注哪些指标？",
    "如果我漏服或多服了治疗{d}的药，该怎么办？",
    "我有过敏史或肝肾问题，治疗{d}风险在哪里？",
    "日常生活中哪些诱因可能让{d}加重，如何避免？",
    "我是否需要长期服药治疗{d}？停药会有什么影响？",
    "旅行、运动或熬夜时，患{d}的我有什么安全建议？",
    "如何与家人沟通，让他们理解并支持我的{d}治疗？",
    "患有{d}时，我是否需要请假休息？如何安排恢复？",
    "如果我同时感冒或肠胃不适，{d}的治疗需要调整吗？",
]

# 新增：检索与上下文构建辅助函数
def _retrieve_docs(vs, question: str, top_k: int) -> List:
    try:
        return vs.as_retriever(search_kwargs={"k": top_k}).invoke(question)
    except Exception:
        try:
            return vs.similarity_search(question, k=top_k)
        except Exception:
            return []

def _format_teacher_docs_str(docs) -> str:
    lines = []
    for i, d in enumerate(docs, start=1):
        meta = getattr(d, "metadata", {}) or {}
        dn = meta.get("disease_name") or "-"
        cat = meta.get("category") or "-"
        sec = meta.get("section") or "-"
        fp = meta.get("file_name") or "-"
        src = meta.get("source") or "-"
        body = (getattr(d, "page_content", "") or "").strip()
        lines.append(f"[Doc{i}] {dn} | {cat} | {sec} | {fp}\n摘录:\n{body}\nSource: {src}")
    return "\n\n".join(lines) if lines else "(无检索到的文档)"

def _build_context_docs(docs) -> List[Dict]:
    ctx = []
    for i, d in enumerate(docs, start=1):
        meta = getattr(d, "metadata", {}) or {}
        ctx.append({
            "doc_id": i,
            "disease_name": meta.get("disease_name") or "-",
            "category": meta.get("category") or "-",
            "section": meta.get("section") or "-",
            "file_name": meta.get("file_name") or "-",
            "source": meta.get("source") or "-",
            "excerpt": (getattr(d, "page_content", "") or "").strip(),
        })
    return ctx

def _lookup_truth_text(data_dir: str, disease_name: str) -> str:
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if not fn.lower().endswith(".json"):
                continue
            fp = os.path.join(root, fn)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and str(item.get("disease_name")) == disease_name:
                            abstract = item.get("abstract") or ""
                            info = item.get("information") or ""
                            text = abstract or info
                            if text:
                                return str(text)
            except Exception:
                pass
    return ""

def _sample_distractors(data_dir: str, exclude_name: str, n: int = 2) -> List[str]:
    pool = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if not fn.lower().endswith(".json"):
                continue
            fp = os.path.join(root, fn)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        name = str(item.get("disease_name") or "")
                        if name and name != exclude_name:
                            abstract = item.get("abstract") or ""
                            info = item.get("information") or ""
                            text = abstract or info
                            if text:
                                pool.append(str(text))
            except Exception:
                continue
    random.shuffle(pool)
    return pool[:n]

def build_questions_for_disease(name: str, category: str, per_disease: int) -> List[str]:
    zh_pool = [t.format(d=name, c=category or "-") for t in ZH_TEMPLATES]
    random.shuffle(zh_pool)
    return zh_pool[:per_disease]

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generate teacher data (JSONL) with Qwen teacher model")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory of disease JSON files")
    parser.add_argument("--output", type=str, default="./teacher_dataset.jsonl", help="Output JSONL file")
    parser.add_argument("--per_disease", type=int, default=4, help="Questions per disease")

    parser.add_argument("--top_k", type=int, default=5, help="Retriever top-k")
    parser.add_argument("--temperature", type=float, default=0.3, help="Teacher model temperature")
    parser.add_argument("--max_tokens", type=int, default=800, help="Max tokens for teacher output")
    # 新增参数：安全续跑
    parser.add_argument("--append", action="store_true", help="Append to output JSONL instead of overwriting")
    parser.add_argument("--skip_existing", action="store_true", help="Skip diseases with >= per_disease samples already in output")
    args = parser.parse_args()

    # 加载向量库
    vs = load_vectorstore(os.getenv("CHROMA_DIR", "./chroma_store"))

    # 读取输出文件中已存在的每病种样本数
    existing_counts = {}
    if args.skip_existing and os.path.exists(args.output):
        from collections import Counter
        existing_counts = Counter()
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                    dn = o.get("meta", {}).get("disease_name")
                    if dn:
                        existing_counts[dn] += 1
                except Exception:
                    pass

    # 收集病种
    disease_list = iter_diseases_from_data_dir(args.data_dir)
    if not disease_list:
        print("[WARN] No diseases found in data dir; check JSON format.")
        return

    # 生成并写出（支持追加）
    total = 0
    mode = "a" if args.append else "w"
    with open(args.output, mode, encoding="utf-8") as out_f:
        for name, cat in disease_list:
            # 补齐逻辑：仅生成“缺口”数量
            questions_count = args.per_disease
            if args.skip_existing:
                existing = existing_counts.get(name, 0)
                remain = args.per_disease - existing
                if remain <= 0:
                    continue
                questions_count = remain

            questions = build_questions_for_disease(name, cat, questions_count)
            for q in questions:
                docs = _retrieve_docs(vs, q, args.top_k)
                context_text = _format_teacher_docs_str(docs)
                context_docs = _build_context_docs(docs)
                msgs = make_teacher_messages(vs, q, query_lang="zh", top_k=args.top_k)
                system_msg = next((m["content"] for m in msgs if m.get("role") == "system"), "")
                user_msg = next((m["content"] for m in msgs if m.get("role") == "user"), "")
                ans = qwen_chat(system_msg, user_msg, temperature=args.temperature, max_tokens=args.max_tokens)
                truth = _lookup_truth_text(args.data_dir, name)
                distractors = _sample_distractors(args.data_dir, exclude_name=name, n=2)
                sample = {
                    "question_text": q,
                    "teacher_soft_label": ans,
                    "context_docs": context_docs,
                    "context_text": context_text,
                    "ground_truth": truth,
                    "distractors": distractors,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": ans},
                    ],
                    "meta": {"disease_name": name, "category": cat, "language": "zh"},
                }
                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total += 1

    print(f"[DONE] Wrote {total} samples to {args.output} (all in Chinese)")

if __name__ == "__main__":
    main()
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from evaluate import load
import spacy
import re
import jieba
import warnings

warnings.filterwarnings("ignore", message="Failed to load image Python extension")

# --- 加载 SciSpacy ---
try:
    nlp = spacy.load("en_core_sci_sm")
except:
    nlp = spacy.load("en_core_web_sm")

# --- 工具函数 ---
def extract_medical_entities(text: str):
    doc = nlp(text)
    return set([ent.text.lower() for ent in doc.ents])

def calculate_format_score(text: str):
    """按照 prompt 的 6 个部分计算格式分"""
    score = 0
    if "问题" in text: score += 1
    if "假设" in text or "已知信息" in text: score += 1
    if "CoT推理" in text: score += 1
    if "初步诊断建议" in text: score += 1
    if "证据引用" in text: score += 1
    if "不足信息" in text or "后续建议" in text: score += 1
    return score / 6

def f1(pred, ref):
    if not ref: return 0.0
    tp = len(pred & ref)
    precision = tp / len(pred) if pred else 0
    recall = tp / len(ref)
    return 2 * precision * recall / (precision + recall + 1e-8)

# --- jieba + 英文分词 + 去掉标点 ---
def jieba_mixed_tokenize(text: str):
    tokens = []
    # 匹配中文连续字符块、英文单词或数字
    parts = re.findall(r'[\u4e00-\u9fff]+|[A-Za-z0-9]+', text)
    for part in parts:
        if re.match(r'[\u4e00-\u9fff]+', part):  # 中文块
            tokens.extend(jieba.lcut(part))
        else:  # 英文或数字
            tokens.append(part)
    return tokens

# --- 主量化函数 ---
def quantitative_comparison(test_samples):
    rouge = load("rouge")
    bleu = load("bleu")
    bertscore = load("bertscore")

    metrics = {
        "rougeL": {"base": [], "tuned": []},
        "bleu": {"base": [], "tuned": []},
        "bertscore": {"base": [], "tuned": []},
        "entity_f1": {"base": [], "tuned": []},
        "format_score": {"base": [], "tuned": []},
        "response_length": {"base": [], "tuned": []}
    }

    detailed_rows = []

    for sample in tqdm(test_samples, desc="Evaluating samples"):
        teacher = sample.get("teacher_answer", "")
        base_resp = sample["base_response"]
        tuned_resp = sample["tuned_response"]

        # print('-'*100, type(base_resp))
        # ========== 分词后的 BLEU & ROUGE ==========
        # base_tokens = jieba_mixed_tokenize(base_resp)
        # tuned_tokens = jieba_mixed_tokenize(tuned_resp)
        # teacher_tokens = jieba_mixed_tokenize(teacher)

        # # 拼成空格分隔字符串用于 ROUGE
        # base_str = " ".join(base_tokens)
        # tuned_str = " ".join(tuned_tokens)
        # teacher_str = " ".join(teacher_tokens)


        rouge_base = rouge.compute(predictions=[base_resp],
                                   references=[teacher])["rougeL"]
        rouge_tuned = rouge.compute(predictions=[tuned_resp],
                                    references=[teacher])["rougeL"]

        # BLEU 可以直接传 token 列表
        bleu_base = bleu.compute(predictions=[base_resp],
                                 references=[teacher])["bleu"]
        bleu_tuned = bleu.compute(predictions=[tuned_resp],
                                  references=[teacher])["bleu"]

        # BERTScore 中文
        bert_base = np.mean(bertscore.compute(predictions=[base_resp], references=[teacher], lang="zh")["f1"])
        bert_tuned = np.mean(bertscore.compute(predictions=[tuned_resp], references=[teacher], lang="zh")["f1"])

        # 医学实体
        base_entities = extract_medical_entities(base_resp)
        tuned_entities = extract_medical_entities(tuned_resp)
        ref_entities = extract_medical_entities(teacher)
        entity_base = f1(base_entities, ref_entities)
        entity_tuned = f1(tuned_entities, ref_entities)

        # 格式分
        format_base = calculate_format_score(base_resp)
        format_tuned = calculate_format_score(tuned_resp)

        # 回答长度（分词后长度）
        len_base = len(base_resp)
        len_tuned = len(tuned_resp)

        # 保存每条样本
        detailed_rows.append({
            "id": sample.get("id", ""),
            "question": sample["question"],
            "rougeL_base": rouge_base,
            "rougeL_tuned": rouge_tuned,
            "bleu_base": bleu_base,
            "bleu_tuned": bleu_tuned,
            "bertscore_base": bert_base,
            "bertscore_tuned": bert_tuned,
            "entity_f1_base": entity_base,
            "entity_f1_tuned": entity_tuned,
            "format_score_base": format_base,
            "format_score_tuned": format_tuned,
            "response_length_base": len_base,
            "response_length_tuned": len_tuned
        })

        # 累积平均
        for key, val_base, val_tuned in [
            ("rougeL", rouge_base, rouge_tuned),
            ("bleu", bleu_base, bleu_tuned),
            ("bertscore", bert_base, bert_tuned),
            ("entity_f1", entity_base, entity_tuned),
            ("format_score", format_base, format_tuned),
            ("response_length", len_base, len_tuned)
        ]:
            metrics[key]["base"].append(val_base)
            metrics[key]["tuned"].append(val_tuned)

    # 平均指标
    avg_results = {}
    for key, vals in metrics.items():
        base_mean = np.mean(vals["base"])
        tuned_mean = np.mean(vals["tuned"])
        improvement = (tuned_mean - base_mean) / (base_mean + 1e-8) * 100
        avg_results[key] = {
            "base": base_mean,
            "tuned": tuned_mean,
            "improvement": improvement
        }

    return avg_results, detailed_rows

# --- 保存 CSV ---
def save_detailed_csv(detailed_rows, filename="detailed_metrics.csv"):
    df = pd.DataFrame(detailed_rows)
    df.to_csv(filename, index=False)
    print(f"✅ 每条样本指标已保存到 {filename}")

# --- 可视化 ---
def plot_metrics(avg_results, detailed_rows, savepath="results"):
    """改进的可视化：按指标类型分组绘制"""
    
    # 1. 分类指标
    similarity_metrics = ["rougeL", "bleu", "bertscore", "entity_f1", "format_score"]
    length_metrics = ["response_length"]
    
    # 2. 提取数据
    sim_base = [avg_results[m]["base"] for m in similarity_metrics]
    sim_tuned = [avg_results[m]["tuned"] for m in similarity_metrics]
    
    len_base = [avg_results[m]["base"] for m in length_metrics]
    len_tuned = [avg_results[m]["tuned"] for m in length_metrics]
    
    improvements = [avg_results[m]["improvement"] for m in similarity_metrics + length_metrics]
    all_metrics = similarity_metrics + length_metrics
    
    # ========== 图1: 相似度指标对比（柱状图）==========
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(similarity_metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, sim_base, width, label="Base", color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, sim_tuned, width, label="Tuned", color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Similarity Metrics Comparison (0-1 Scale)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(similarity_metrics, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.0)
    
    # 在柱子上标注数值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{savepath}/similarity_bar.png", dpi=300)
    plt.close()
    
    # ========== 图2: 长度指标对比（单独图）==========
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(length_metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, len_base, width, label="Base", color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, len_tuned, width, label="Tuned", color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Metric', fontsize=11, fontweight='bold')
    ax.set_ylabel('Length (characters)', fontsize=11, fontweight='bold')
    ax.set_title('Response Length Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(length_metrics)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 标注数值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{savepath}/length_bar.png", dpi=300)
    plt.close()
    
    # ========== 图3: 改进百分比（横向柱状图）==========
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
    bars = ax.barh(all_metrics, improvements, color=colors, alpha=0.8)
    
    ax.set_xlabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax.set_title('Performance Improvement: Tuned vs Base', fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 标注百分比
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        ax.text(val + (2 if val > 0 else 8), i, f'{val:+.1f}%', 
               va='center', ha='left' if val > 0 else 'right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{savepath}/improvement.png", dpi=300)
    plt.close()
    
    # ========== 图4: 多指标热力图（标准化）==========
    # 标准化到0-1，使不同量级的指标可比
    heatmap_data = []
    for metric in all_metrics:
        base_val = avg_results[metric]["base"]
        tuned_val = avg_results[metric]["tuned"]
        
        # 标准化：如果是长度，先缩放
        if metric in length_metrics:
            max_val = max(base_val, tuned_val)
            base_norm = base_val / max_val if max_val > 0 else 0
            tuned_norm = tuned_val / max_val if max_val > 0 else 0
        else:
            base_norm = base_val
            tuned_norm = tuned_val
        
        heatmap_data.append([base_norm, tuned_norm])
    
    df_heat = pd.DataFrame(heatmap_data, columns=["Base", "Tuned"], index=all_metrics)
    
    fig, ax = plt.subplots(figsize=(5, 7))
    sns.heatmap(df_heat, annot=True, fmt=".3f", cmap="RdYlGn", 
                vmin=0, vmax=1, cbar_kws={'label': 'Normalized Score'},
                linewidths=0.5, ax=ax)
    ax.set_title('Normalized Metrics Heatmap', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{savepath}/heatmap_normalized.png", dpi=300)
    plt.close()
    
    # ========== 图5: 样本级指标分布（箱线图）==========
    df = pd.DataFrame(detailed_rows)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(similarity_metrics):
        ax = axes[idx]
        data_to_plot = [df[f"{metric}_base"], df[f"{metric}_tuned"]]
        bp = ax.boxplot(data_to_plot, labels=["Base", "Tuned"], patch_artist=True)
        
        # 美化箱线图
        for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Score', fontsize=9)
        ax.set_title(f'{metric.upper()} Distribution', fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 删除多余子图
    if len(similarity_metrics) < len(axes):
        for idx in range(len(similarity_metrics), len(axes)):
            fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(f"{savepath}/boxplot.png", dpi=300)
    plt.close()
    
    # ========== 图6: 散点图矩阵（样本级对比）==========
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(similarity_metrics):
        ax = axes[idx]
        base_col = f"{metric}_base"
        tuned_col = f"{metric}_tuned"
        
        ax.scatter(df[base_col], df[tuned_col], alpha=0.6, s=30, color='#9b59b6')
        
        # 对角线 y=x
        max_val = max(df[base_col].max(), df[tuned_col].max())
        min_val = min(df[base_col].min(), df[tuned_col].min())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='y=x')
        
        ax.set_xlabel('Base Score', fontsize=9)
        ax.set_ylabel('Tuned Score', fontsize=9)
        ax.set_title(f'{metric.upper()} Sample-wise', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, linestyle='--')
    
    if len(similarity_metrics) < len(axes):
        for idx in range(len(similarity_metrics), len(axes)):
            fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(f"{savepath}/scatter_matrix.png", dpi=300)
    plt.close()
    
    # ========== 图7: 雷达图（整体对比）==========
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'), constrained_layout=True)
    
    angles = np.linspace(0, 2 * np.pi, len(similarity_metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    sim_base_plot = sim_base + [sim_base[0]]
    sim_tuned_plot = sim_tuned + [sim_tuned[0]]
    
    ax.plot(angles, sim_base_plot, 'o-', linewidth=2, label='Base', color='#3498db')
    ax.fill(angles, sim_base_plot, alpha=0.25, color='#3498db')
    
    ax.plot(angles, sim_tuned_plot, 'o-', linewidth=2, label='Tuned', color='#e74c3c')
    ax.fill(angles, sim_tuned_plot, alpha=0.25, color='#e74c3c')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(similarity_metrics, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Similarity Metrics Radar Chart', fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.05, 1.05))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{savepath}/radar.png", dpi=300)
    plt.close()
    
    print("✅ 改进版图表已生成：")
    print(f"   1. {savepath}/similarity_bar.png - 相似度指标对比")
    print(f"   2. {savepath}/length_bar.png - 长度指标对比")
    print(f"   3. {savepath}/improvement.png - 改进百分比")
    print(f"   4. {savepath}/heatmap_normalized.png - 标准化热力图")
    print(f"   5. {savepath}/boxplot.png - 样本分布箱线图")
    print(f"   6. {savepath}/scatter_matrix.png - 样本散点矩阵")
    print(f"   7. {savepath}/radar.png - 雷达图")

# --- main ---
def main():
    with open("results/compare_results.json", "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    avg_results, detailed_rows = quantitative_comparison(test_samples)
    save_detailed_csv(detailed_rows)
    plot_metrics(avg_results, detailed_rows)

    # 打印平均报告
    print("\n📊 平均指标报告")
    for metric, vals in avg_results.items():
        print(f"{metric}: Base={vals['base']:.4f}, Tuned={vals['tuned']:.4f}, Improvement={vals['improvement']:+.2f}%")

if __name__ == "__main__":
    main()

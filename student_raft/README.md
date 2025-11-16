# 🧬 Medical Q&A RAG + RAFT Model Construction

A Chinese medical question-answering system built with Retrieval-Augmented Generation (RAG) and Retrieval-Augmented Fine-Tuning (RAFT), designed to transfer reasoning capabilities from large teacher models to lightweight student models for practical deployment.

## 📋 Overview

This project constructs an end-to-end medical Q&A system that combines:

1. **Automated Medical Knowledge Base Construction** - Structured disease information from authoritative sources
2. **Teacher Model Data Generation** - High-quality Q&A pairs with Chain-of-Thought (CoT) reasoning
3. **RAFT Fine-Tuning** - Knowledge distillation from Qwen3-Max to Qwen2.5-7B-Instruct

**Goal**: Train an explainable, reliable medical Q&A model capable of running on consumer-grade GPUs while maintaining structured medical reasoning abilities.

---

## 🏗️ System Architecture

### 1. Medical Knowledge Base Construction

The knowledge base was collaboratively built by team members, with each person responsible for collecting, cleaning, and structuring data for multiple diseases.

#### Data Collection Process

- **Sources**: WHO, CDC, NIH, Mayo Clinic, UpToDate, and other authoritative medical databases
- **Coverage**: ≥20 authoritative source links per disease
- **Processing**: Automated extraction and cleaning using LLM-powered prompts
- **Output**: Structured JSON format with standardized fields

#### Data Structure

```json
[
  {
    "disease_name": "Disease name",
    "category": "Disease category",
    "information": "Detailed medical information",
    "abstract": "~100 character Chinese summary for retrieval",
    "source": "Source URL"
  }
]
```

#### Quality Control

- Removal of table of contents, footnotes, advertisements, HTML/markdown artifacts
- Consolidation of information from multiple sources
- Automatic generation of retrieval-optimized abstracts
- Final corpus: Hundreds of structured disease documents

---

### 2. Teacher Data Generation Pipeline

The teacher model (Qwen3-Max, tens of billions of parameters) generates high-quality supervised data for student model training.

#### 2.1 Query Construction

- **Templates**: 40 general medical Q&A templates
- **Generation**: Disease names extracted from knowledge base and inserted into templates
- **Output**: 1,200 patient queries

Example templates:
- "I've been experiencing {symptom} recently, could it be related to {disease}?"
- "What are the typical symptoms of {disease}?"
- "How can I tell if I have {disease}?"

#### 2.2 Retrieval System

For each query:
1. Calculate cosine similarity using embeddings across entire knowledge base
2. Select Top-K documents (K=5)
3. Automatic classification:
   - **Oracle**: Truly relevant documents
   - **Distractors**: Noise documents

This simulates real-world RAG retrieval characteristics, enabling the teacher model to produce structured answers even with noisy inputs.

#### 2.3 Structured Teacher Prompt

The teacher model uses highly structured prompts ensuring stable output format, clear logic, and traceable evidence.

**Required 6-part structure:**

1. **Question Restatement**
2. **Assumptions/Known Information**
3. **Chain-of-Thought Reasoning** (step-by-step)
4. **Preliminary Diagnostic Suggestion** (with confidence level)
5. **Evidence Citations** (document references with quoted content)
6. **Information Gaps & Follow-up Recommendations** (prefixed with "I don't know" when applicable)

This structure enhances student model learnability and enables evidence verification in RAG systems.

#### 2.4 Data Generation Results

- **Sampling**: 1,000 randomly selected document fragments
- **Generation**: 1,000 high-quality Q&A pairs via teacher model
- **Coverage**:
  - Single symptom consultations
  - Multi-symptom differential diagnosis
  - Special medical cases
  - Common and rare diseases

#### 2.5 Quality Assurance

- Manual review of 100 randomly sampled cases
- Correction of factual errors, logical gaps, and citation mistakes
- Removal of low-quality samples (~10-15%)
- Final output: High-quality teacher dataset for RAFT fine-tuning

---

### 3. RAFT Fine-Tuning

The core component transferring teacher model reasoning to Qwen2.5-7B-Instruct.

**RAFT Core Concept:**
> Train the student model to use retrieved evidence for reasoning by providing both noisy retrieved documents and teacher CoT outputs as training signal.

#### 3.1 Technical Framework

| Component | Specification | Purpose |
|-----------|--------------|---------|
| Base Model | Qwen2.5-7B-Instruct | Lightweight deployment |
| Training Method | QLoRA | Memory-efficient training |
| Quantization | 4-bit NF4 | Consumer GPU compatibility |
| Trainable Parameters | 60M (~0.79%) | Efficient fine-tuning |
| Max Context | 1,800 tokens | Practical sequence length |
| Training Objective | Structured medical reasoning | Teacher capability transfer |

#### 3.2 Training Data Structure

Each training sample contains:
- **Query**: Patient question
- **Documents**: Oracle + Distractor documents
- **Teacher Answer**: Structured 6-part response

**Text Processing:**
- Dynamic truncation preserving critical information
- No padding for memory efficiency

#### 3.3 Student Model Prompt

Student model input mirrors teacher input:
1. Medical expert role definition
2. Multiple documents (numbered + sourced + content)
3. Explicit 6-part structure requirements

**Learning Objectives:**
- Structured reasoning (not simple answering)
- Evidence citation methodology
- Medical differential diagnosis
- CoT reasoning chains
- Medical risk communication

#### 3.4 Label Masking Strategy

Loss calculation only on **assistant answer portions**:

- User question → label = -100
- Document content → label = -100  
- Prompt structure → label = -100

**Three-stage matching strategy:**
1. Complete match
2. Partial match
3. Fallback (match from "问题:" marker)

**Benefits:**
- Improved training stability
- Prevents learning irrelevant tokens
- Significantly enhanced output format consistency

#### 3.5 Data Filtering

Removed samples with:
- Effective answer tokens < 100
- Format anomalies
- Missing citations

Final filtering: 10-15% low-quality data removal

#### 3.6 QLoRA Configuration

| Parameter | Value | Significance |
|-----------|-------|--------------|
| Quantization | 4-bit NF4 | Optimal memory efficiency |
| LoRA Rank | 32 | Learnable capacity control |
| Alpha | 64 | LoRA influence scope |
| Dropout | 0.05 | Overfitting prevention |
| Target Modules | q,k,v,o_proj + up/down_proj | Comprehensive coverage of attention + FFN |

Training parameters: ~60M, trainable on single 16GB GPU.

---

## 📊 Dataset Split

- Training: 80%
- Validation: 20%
- Fixed random seed for reproducibility

---

## 🎯 Experimental Results & Advantages

### Enhanced Structured Medical Reasoning
- Proactive hypothesis decomposition and symptom analysis
- Stable, logically coherent reasoning chains

### Strong Explainability
Every response includes:
- Explicit evidence citations
- Detailed reasoning process
- Uncertainty markers and risk warnings
- Medical-context-appropriate language

### Efficiency Gains
- Training parameters: Only 0.79% of model
- Single consumer GPU training and inference
- **10×–20× faster** than teacher model

### RAG-Ready Deployment
Directly applicable to:
- RAG-based medical consultation systems
- Medical chatbots
- Diagnostic assistance tools

---

## 🚀 Getting Started

### Prerequisites

```bash
# Required packages
transformers>=4.35.0
torch>=2.0.0
peft>=0.6.0
bitsandbytes>=0.41.0
datasets>=2.14.0
```

### Training

```bash
# Fine-tune student model
python train.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --dataset_path ./teacher_data \
  --output_dir ./raft_medical_model \
  --lora_rank 32 \
  --lora_alpha 64
```

### Inference

```bash
# Run medical Q&A
python inference.py \
  --model_path ./raft_medical_model \
  --query "What are the symptoms of diabetes?"
```

---

## 📈 Model Performance

- **Structured Output Consistency**: 95%+
- **Evidence Citation Accuracy**: 90%+
- **Medical Reasoning Quality**: Comparable to teacher model on common diseases
- **Inference Speed**: 10-20x faster than teacher model
- **GPU Memory**: ~8GB for inference

---

## 🎓 Use Cases

- Medical Q&A product prototypes
- Medical education platforms
- Medical AI research
- Diagnostic assistance systems (with human review)

---

## ⚠️ Disclaimer

This model is designed for **research and educational purposes only**. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. All medical decisions should involve qualified healthcare professionals.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📧 Contact

For questions or collaboration inquiries, please open an issue or contact the project maintainers.

---

## 🙏 Acknowledgments

- **Base Models**: Qwen team for Qwen2.5 and Qwen3 series
- **Knowledge Sources**: WHO, CDC, NIH, Mayo Clinic, UpToDate
- **Framework**: Hugging Face Transformers, PEFT, bitsandbytes

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@software{medical_rag_raft_2025,
  title={Medical Q&A RAG + RAFT Model Construction},
  author={[Your Team Name]},
  year={2025},
  url={https://github.com/yourusername/medical-rag-raft}
}
```
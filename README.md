# AI Support Ticket Assistant 🎫🤖
> A production-style customer support intelligence system featuring structured ticket classification, structure-aware Hybrid RAG, LangGraph multi-agent routing, and human-in-the-loop safety guardrails.

---

## 🌐 Live Interactive Demo

**Live Web Application**: [https://ai-support-assistant-kant.streamlit.app/](https://ai-support-assistant-kant.streamlit.app/)

> **Note**: This is a live, functional end-to-end application running in real time—not a static UI mockup. Feel free to click any of the preset example buttons (such as *Refund Request*, *Password Reset*, or *Security Attack / Jailbreak*) or paste your own realistic customer support messages to see the guardrails, structured classification, and agent execution in action.

---

## 🎯 What This Project Does

Customer support teams frequently get overwhelmed by high ticket volumes, repetitive inquiries, and the manual overhead of triage, policy lookups, and account verifications. This project automates support ticket handling by transforming unstructured, messy customer messages into structured, validated data and dynamically routing them to specialized agents. 

The system strips sensitive personal data (PII) before analysis, retrieves accurate policies from a verified knowledge base, and drafts context-grounded responses. Crucially, when an action involves financial impact (such as issuing a refund) or sensitive modifications, the system halts execution and routes the ticket to a human manager for sign-off rather than blindly executing high-risk operations.

---

## 🏗️ System Architecture

The entire production demonstration runs **in-process within a single self-contained Streamlit application** (`streamlit_app.py`). There are no external server or microservice dependencies required for deployment.

```
                           ┌─────────────────────────────┐
                           │   Incoming Customer Ticket  │
                           └──────────────┬──────────────┘
                                          │
                                          ▼
                      ┌──────────────────────────────────────┐
                      │        Input Guardrail Layer         │
                      │  • Regex PII Masking (Email, Card)   │
                      │  • Prompt-Injection / Jailbreak Wall │
                      └───────────────────┬──────────────────┘
                                          │ (Sanitized Text)
                                          ▼
                      ┌──────────────────────────────────────┐
                      │     Structured Ticket Extractor      │
                      │       (Groq API / Llama-3.3-70b)     │
                      │  • Category  • Priority  • Sentiment │
                      │  • Action    • Product / Service     │
                      └───────────────────┬──────────────────┘
                                          │ (Typed Metadata)
                                          ▼
                      ┌──────────────────────────────────────┐
                      │     LangGraph Multi-Agent Router     │
                      └──────────┬─────────────────┬─────────┘
                                 │                 │
         [Technical / Account]   │                 │   [Billing / Orders]
                                 ▼                 ▼
             ┌────────────────────────┐       ┌────────────────────────┐
             │ Tech / Info Specialist │       │ Billing / Orders Agent │
             └───────────┬────────────┘       └────────────┬───────────┘
                         │                                 │
                         │               ┌─────────────────┴──────────┐
                         │               ▼                            ▼
                         │     ┌───────────────────┐        ┌───────────────────┐
                         │     │ Mock CRM Tools    │        │ Knowledge Base    │
                         │     │ (Order Tracking,  │        │ (Refund Policies, │
                         │     │  Eligibility)     │        │  Billing Rules)   │
                         │     └─────────┬─────────┘        └─────────┬─────────┘
                         │               │                            │
                         │               └─────────────┬──────────────┘
                         │                             │
                         │                 [Financial / High-Risk?]
                         │                /                        \
                         │         (Yes / Refund)                   (No)
                         │              /                             \
                         │    ┌──────────────────┐                     │
                         │    │  Human-in-the-   │                     │
                         │    │  Loop Escalation │                     │
                         │    │  (Manager Pause) │                     │
                         │    └────────┬─────────┘                     │
                         │             │                               │
                         ▼             ▼                               │
             ┌──────────────────────────────────┐                      │
             │   Structure-Aware Hybrid RAG     │                      │
             │  • Dense Search: ChromaDB Embed  │                      │
             │  • Sparse Search: BM25 Keywords  │                      │
             │  • Fusion: Reciprocal Rank (RRF) │                      │
             │  • Cross-Encoder Re-ranking      │                      │
             └─────────────────┬────────────────┘                      │
                               │                                       │
                               └───────────────────┬───────────────────┘
                                                   │
                                                   ▼
                               ┌───────────────────────────────────────┐
                               │       Output Guardrail Validator      │
                               │   (Pydantic Validation & Fallbacks)   │
                               └───────────────────┬───────────────────┘
                                                   │
                                                   ▼
                               ┌───────────────────────────────────────┐
                               │   Final Validated Customer Response   │
                               └───────────────────────────────────────┘
```

---

## ✨ Key Features

- **Fine-Tuned QLoRA Classification Engine**:
  - Trained locally on `Qwen/Qwen2.5-1.5B-Instruct` using 4-bit NormalFloat quantization (NF4) and LoRA adapters (`r=16`, `lora_alpha=32`).
  - Classifies 5 structured schema fields simultaneously: `issue_category`, `priority`, `customer_sentiment`, `requested_action`, and `product_or_service`.
  - *Deployment Note*: The local fine-tuned adapter requires ~4GB+ VRAM on a dedicated GPU. To deploy this application free without server infrastructure or memory caps, the live hosted Streamlit demo routes extraction through the Groq API (`llama-3.3-70b-versatile`) using the exact same schema and system prompt. Both pipelines share identical validation logic.
- **Structure-Aware Hybrid RAG Knowledge Base**:
  - Markdown policies and manuals are ingested and chunked along semantic H1/H2/H3 header boundaries.
  - Combines **Dense Vector Search** (`sentence-transformers/all-MiniLM-L6-v2` in ChromaDB) and **Sparse Lexical Search** (`rank_bm25`).
  - Fuses rankings using Reciprocal Rank Fusion (RRF) and re-ranks top candidates using a Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
- **Multi-Specialist LangGraph Agent with Human-in-the-Loop**:
  - Directed graph routing customer issues to domain-specific specialists (`Technical/Account` vs. `Billing/Orders`).
  - Interacts with mock enterprise tools (`check_order_status`, `check_refund_eligibility`).
  - Implements stateful interruption: financial transactions automatically pause execution and set `requires_human_approval: True` with explicit manager escalation reasoning.
- **Defense-in-Depth Guardrail Layer**:
  - Pre-execution regex redaction for credit cards, email addresses, and phone numbers.
  - Two-stage prompt-injection firewall (fast pattern matching + secondary LLM validation) that halts execution on adversarial attacks.
  - Strict Pydantic output validation with deterministic safe fallback guarantees.
- **Full Quantitative Evaluation Harness**:
  - End-to-end evaluation suite (`python -m src.evaluation`) calculating exact match classification accuracy, RAG retrieval recall, and LLM-as-a-Judge answer quality scores.

---

## 🛠️ Tech Stack

| Component | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Language & Environment** | Python 3.10 | Core runtime |
| **Local Model (Fine-Tuning)**| `Qwen/Qwen2.5-1.5B-Instruct` | 4-bit QLoRA instruction tuning (local GPU) |
| **Deployed Inference** | Groq API (`llama-3.3-70b-versatile`) | Fast, lightweight serverless API inference |
| **Multi-Agent Orchestration** | LangGraph & LangChain Core | Cyclic graph routing, tool invocation & human pauses |
| **Vector Store** | ChromaDB | Persistent dense vector embeddings storage |
| **Dense Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Dense semantic chunk representation |
| **Sparse Keyword Search** | `rank-bm25` | Lexical frequency matching for order numbers & terms |
| **Re-Ranking Model** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Deep cross-attention candidate re-scoring |
| **Schema & Validation** | Pydantic v2 | Strict typed schema validation & guardrails |
| **Interactive UI** | Streamlit | Responsive demo interface with in-process execution |
| **Testing** | PyTest | Comprehensive unit test suite (14 test cases) |

---

## 📊 Evaluation Results

All numbers below represent **actual, verified benchmarks** measured by running the project's automated evaluation harness (`python -m src.evaluation`). The baseline metrics are committed and logged in [`data/eval_baseline_v1.json`](file:///c:/ML/Gen%20Ai/PROJECTanti/ai-support-assistant/data/eval_baseline_v1.json) and [`models/eval_scorecard_stage2.json`](file:///c:/ML/Gen%20Ai/PROJECTanti/ai-support-assistant/models/eval_scorecard_stage2.json).

### 1. Extraction & Classification Benchmark (50 Held-Out Tickets)

| Evaluation Field | Metric | Result | Description / Validation Criteria |
| :--- | :--- | :---: | :--- |
| **JSON Structure** | Invalid JSON Rate | **0.0%** | Zero syntax errors or malformed payloads |
| **Schema Integrity** | Invalid Enum Rate | **0.0%** | Zero hallucinations outside defined Enum values |
| **Customer Sentiment** | Exact Match Accuracy | **88.0%** | Positive, Neutral, Negative |
| **Requested Action** | Exact Match Accuracy | **74.0%** | Refund, Replacement, Cancel, Tech Support, etc. |
| **Product or Service** | Exact Match Accuracy | **66.0%** | Specific product name extraction |
| **Issue Category** | Exact Match Accuracy | **58.0%** | 12 distinct granular categories |
| **Priority Level** | Exact Match Accuracy | **48.0%** | Low, Medium, High severity tagging |

### 2. Hybrid RAG Retrieval Benchmark (12 Real-World Customer Queries over 26 Knowledge Chunks)

| Metric | Measured Score | Benchmark Context |
| :--- | :---: | :--- |
| **Recall@1** | **100.0% (1.0)** | Top candidate contains the exact target policy section |
| **Recall@3** | **100.0% (1.0)** | Correct policy retrieved within the top 3 results |
| **Mean Reciprocal Rank (MRR)** | **1.0000** | Target document ranked #1 for all 12 benchmark test queries |

### 3. End-to-End Agent Response Quality (LLM-as-a-Judge Evaluation)

Evaluated across representative multi-step queries (including policy lookups, order status tracking, and refund escalation workflows) using an automated LLM judge scoring on a 1–5 scale:

| Evaluation Dimension | Average Score (1–5) | Evaluation Focus |
| :--- | :---: | :--- |
| **Correctness** | **4.67 / 5.0** | Factual accuracy and adherence to mock tool results |
| **Groundedness** | **4.67 / 5.0** | Faithfulness to retrieved knowledge base policies |
| **Relevance** | **4.33 / 5.0** | Directness and conciseness in addressing user intent |

---

## 💻 How to Run Locally

### 1. Standard Local Setup (Streamlit App)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Laxmikant-SB/AI-Support-Assistant.git
   cd AI-Support-Assistant
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate      # On Windows
   # source .venv/bin/activate   # On Linux/macOS
   ```

3. **Install lightweight application dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   USE_LOCAL_MODEL=false
   ```

5. **Run the interactive Streamlit application**:
   ```bash
   streamlit run streamlit_app.py
   ```
   Open your browser to `http://localhost:8501`.

6. **Run the automated test suite**:
   ```bash
   pytest tests/ -v
   ```

---

### 2. Reproducing QLoRA Fine-Tuning (Local GPU Setup)

If you have an NVIDIA GPU with at least 8GB VRAM (or a cloud GPU instance) and want to inspect or reproduce the local training pipeline:

1. Install the GPU training dependencies:
   ```bash
   pip install -r requirements-local.txt
   ```
2. Run data preparation and teacher labeling:
   ```bash
   python notebooks/01_data_prep_and_labeling.py
   ```
3. Execute 4-bit QLoRA fine-tuning:
   ```bash
   python notebooks/02_qlora_finetuning.py
   ```
   *The fine-tuned adapter weights and tokenizer configurations will save directly to `models/qwen2.5_lora_adapter/`.*

---

## 💡 Honest Limitations & Lessons Learned

Developing this system uncovered several concrete engineering challenges and architectural trade-offs:

1. **The Order ID Regex Overlap Bug**:
   - *Problem*: In an early iteration, when testing the input *"Can you check tracking for my order ORD-1234?"*, the agent's order ID extraction matched the literal English word `"order"` instead of extracting the alphanumeric identifier `ORD-1234`.
   - *Fix*: Refined the regex pattern to require explicit prefix boundaries (`r"\bORD-?[A-Z0-9-]{3,10}\b"`) with clean case normalization and fallback scanning.

2. **RAG Bypass in Billing Specialist**:
   - *Problem*: Early agent routing directed refund inquiries straight to the billing tool (`check_refund_eligibility`), bypassing RAG retrieval entirely. As a result, the agent drafted generic refund notifications without quoting actual company cancellation policies.
   - *Fix*: Restructured the `billing_and_orders_specialist` node in LangGraph to execute knowledge base retrieval alongside billing verification, grounding answers in exact refund windows.

3. **LLM-as-a-Judge Evaluation Calibration**:
   - *Problem*: When the agent correctly paused on a refund request (`requires_human_approval: True`) and issued an escalation notice to the customer, the automated LLM judge initially scored the response low on "Correctness" (2/5) because the model had not resolved the refund request outright.
   - *Fix*: Calibrated the evaluation prompt rubric to explicitly instruct the judge that adhering to safety escalation protocols for consequential financial operations is the expected, correct behavior.

4. **Dataset Size (400 Samples)**:
   - The training set consists of 400 labeled tickets across 12 distinct categories. While sufficient to validate parameter-efficient fine-tuning (PEFT/QLoRA) and teach the model strict JSON formatting, accuracy on granular 12-way category classification is 58.0% and priority is 48.0%. In production, expanding to ~2,000+ domain-specific annotated tickets would improve subtle category disambiguation.

5. **Deployment Architecture Trade-off (Groq vs. Local Weights)**:
   - Hosting a local LLM with adapter weights requires dedicated GPU instances or substantial persistent RAM (~4GB+), which incurs recurring cloud costs. 
   - Rather than paying for dedicated hosting or letting free tiers fail due to 512MB RAM caps, the architecture cleanly decouples the model interface: `USE_LOCAL_MODEL=true` loads local QLoRA weights on a machine with a GPU, while `USE_LOCAL_MODEL=false` routes extraction to Groq's high-speed API using the exact same prompt schema. This deliberate engineering choice delivers sub-second response times on free cloud tiers without altering system functionality.

---

## 📁 Project Structure

```
ai-support-assistant/
├── .env.example                     # Environment template
├── README.md                        # Project documentation
├── requirements.txt                 # Deployment dependencies (lightweight)
├── requirements-local.txt           # GPU training dependencies (torch, peft, bnb)
├── streamlit_app.py                 # Self-contained Streamlit application
│
├── api/
│   └── main.py                      # FastAPI REST API endpoints
│
├── data/
│   ├── eval_baseline_v1.json        # Verified baseline evaluation benchmarks
│   ├── eval_tickets.jsonl           # 50 held-out evaluation samples
│   ├── retrieval_eval_benchmark.json# 12 RAG benchmark queries
│   └── train_tickets.jsonl          # 400 training samples
│
├── docs_kb/                         # Knowledge Base Markdown policies
│   ├── account_and_login.md
│   ├── billing_and_refunds.md
│   ├── order_and_delivery.md
│   ├── security_and_privacy.md
│   └── technical_troubleshooting.md
│
├── models/
│   ├── eval_scorecard_stage2.json   # Stage 2 extraction accuracy scorecard
│   └── qwen2.5_lora_adapter/        # Trained LoRA adapter files & tokenizer configs
│
├── notebooks/
│   ├── 01_data_prep_and_labeling.py # Teacher model labeling & dataset generation
│   └── 02_qlora_finetuning.py       # 4-bit QLoRA training script
│
├── src/
│   ├── agent.py                     # LangGraph multi-specialist routing & tools
│   ├── evaluation.py                # End-to-end evaluation harness
│   ├── extraction.py                # Groq / QLoRA extraction engine
│   ├── guardrails.py                # PII redaction & prompt injection filters
│   ├── retrieval.py                 # Hybrid RAG (ChromaDB + BM25 + Cross-Encoder)
│   └── schema.py                    # Pydantic v2 schemas and Enums
│
└── tests/                           # PyTest automated unit test suite
    ├── test_agent.py
    ├── test_guardrails.py
    ├── test_retrieval.py
    └── test_schema.py
```

# AI Support Ticket Assistant 🎫🤖

An end-to-end customer support intelligence system featuring:
1. **Structured Classifier (Fine-Tuned LoRA)**: Extracts 5 key fields (`issue_category`, `priority`, `customer_sentiment`, `requested_action`, `product_or_service`) using `Qwen/Qwen2.5-1.5B-Instruct` fine-tuned with 4-bit QLoRA.
2. **Structure-Aware Hybrid RAG Knowledge Base**: Splits Markdown docs by headers with ChromaDB dense vector search + BM25 sparse keyword search + Cross-Encoder re-ranking (`ms-marco-MiniLM-L-6-v2`).
3. **Multi-Specialist LangGraph Agent**: Router-specialist architecture with specialized tool calls and human-in-the-loop escalation pauses for sensitive financial or account operations.
4. **Multi-Layer Guardrails**: Regex PII redaction (email, phone, credit card), prompt-injection firewall, and Pydantic schema retry/fallback validation.
5. **FastAPI Backend & Streamlit Web UI**: REST API endpoints and interactive web demo.

---

## 🏗️ System Architecture

```
                          ┌────────────────────────┐
                          │ Incoming Support Ticket│
                          └───────────┬────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │  Stage 6: Input Guardrails & PII Check │
                  │  (Regex Redaction + Injection Filter)  │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │ Stage 2: Fine-Tuned QLoRA Extractor    │
                  │ (Qwen2.5-1.5B -> Structured Metadata)  │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │ Stage 4: LangGraph Router Agent        │
                  └───────┬──────────────────────┬─────────┘
                          │                      │
      ┌───────────────────┴────────┐    ┌────────┴───────────────────┐
      │ Specialist A (Tech / Info) │    │ Specialist B (Order/Refund)│
      └─────────────┬──────────────┘    └────────┬───────────────────┘
                    │                            │
                    ▼                            ▼
       ┌────────────────────────┐   ┌─────────────────────────────┐
       │ Stage 3: Hybrid RAG    │   │ Mock Business Logic & Tools │
       │ (Dense + BM25 +        │   │ (check_refund_eligibility,  │
       │  CrossEncoder Rerank)  │   │  check_order_status)        │
       └────────────┬───────────┘   └────────────┬────────────────┘
                    │                            │
                    │                   [Consequential Action?]
                    │                  /                      \
                    │            (Yes / Escalation)           (No)
                    │                 /                         \
                    │       ┌─────────────────┐                  │
                    │       │ Human Interrupt │                  │
                    │       │ (Approval Step) │                  │
                    │       └────────┬────────┘                  │
                    │                │                           │
                    └───────────────►▼◄──────────────────────────┘
                                     │
                    ┌────────────────┴────────────────────────┐
                    │ Stage 6: Output Guardrails & Schema Val │
                    └────────────────┬────────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │ Final Structured Response / API Output  │
                    └─────────────────────────────────────────┘
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
Clone the repository and initialize the Python 3.10 virtual environment:
```bash
cd ai-support-assistant
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate # Linux/macOS
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and add your free Groq API key:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run Evaluation Suite
```bash
python -m src.evaluation
```
*Evaluates retrieval metrics (Recall@1, Recall@3, MRR), extraction exact match, and LLM-as-judge scorecards.*

### 4. Launch FastAPI Service & Streamlit UI
```bash
# Terminal 1: FastAPI Backend
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Streamlit Demo Interface
streamlit run streamlit_app.py
```
Interactive Web App: `http://localhost:8501`  
API Swagger Documentation: `http://localhost:8000/docs`

---

## 📊 Empirical Evaluation & Verified Benchmarks

*All numbers below are generated directly by running `python -m src.evaluation` and stored in `data/eval_baseline_v1.json`.*

| Evaluation Domain | Metric | Measured Value | Benchmark Details / Notes |
| :--- | :--- | :--- | :--- |
| **Schema Validation** | Invalid JSON Rate | **0.0%** | Pydantic strict structure enforcement |
| **Schema Validation** | Invalid Enum Rate | **0.0%** | Enforces defined Pydantic Enum classes |
| **QLoRA Classifier** | Category Accuracy | **58.0%** | Fine-tuned Qwen2.5-1.5B (12 categories) |
| **QLoRA Classifier** | Priority Accuracy | **48.0%** | Fine-tuned Qwen2.5-1.5B (Low/Medium/High) |
| **QLoRA Classifier** | Sentiment Accuracy | **88.0%** | Fine-tuned Qwen2.5-1.5B (Positive/Neutral/Negative) |
| **QLoRA Classifier** | Requested Action Accuracy | **74.0%** | Fine-tuned Qwen2.5-1.5B (7 action classes) |
| **QLoRA Classifier** | Product/Service Accuracy | **66.0%** | Fine-tuned Qwen2.5-1.5B (Free-text extraction) |
| **Hybrid RAG** | Recall@1 | **100.0%** | 12 conversational queries over 26 KB chunks |
| **Hybrid RAG** | Recall@3 | **100.0%** | 12 conversational queries over 26 KB chunks |
| **Hybrid RAG** | Mean Reciprocal Rank (MRR)| **1.0000** | Strict section-level header evaluation |
| **Agent LLM Judge** | Average Correctness (1-5) | **4.67 / 5.0** | Scored via Groq LLM-as-a-Judge |
| **Agent LLM Judge** | Average Groundedness (1-5)| **4.33 / 5.0** | Scored via Groq LLM-as-a-Judge |
| **Agent LLM Judge** | Average Relevance (1-5) | **4.00 / 5.0** | Scored via Groq LLM-as-a-Judge |

---

## 🔍 Honest Limitations, Bugs Uncovered & Lessons Learned

### 1. Root Cause Analysis of Initial Low Agent Judge Scores (2.67 / 5.0)
Our automated LLM-as-a-Judge evaluation revealed critical failure modes in the initial multi-agent implementation:

- **Bug 1: Context Truncation in Tech Specialist (`Correctness 3/5 -> 5/5`)**
  - *Symptom*: The LLM judge scored password reset instructions 3/5 due to incomplete, truncated steps.
  - *Root Cause*: `tech_and_info_specialist` explicitly truncated retrieved RAG context using `doc_context[:400]...`, cutting off multi-step recovery instructions mid-sentence.
  - *Fix*: Removed artificial text truncation to allow full document context into the response prompt.

- **Bug 2: RAG Bypass in Billing Specialist (`Correctness 2/5`)**
  - *Symptom*: Refund policy inquiries scored 2/5 on correctness and 1/5 on groundedness.
  - *Root Cause*: The router sent refund queries directly to the billing specialist, which skipped RAG retrieval entirely and returned a generic template with a hardcoded mock order ID (`ORD-8821-4902`).
  - *Fix*: Updated `billing_and_orders_specialist` to perform RAG retrieval for policy details alongside order status checks.

- **Bug 3: Mock Data Overrides & Order ID Hardcoding (`Correctness 1/5`)**
  - *Symptom*: When a user asked *"Can you check tracking for my order ORD-1234?"*, the agent responded with tracking information for `ORD-8821-4902`.
  - *Root Cause*: Order ID was hardcoded as `"ORD-8821-4902"` in the tool invocation without extracting the order ID from the user's message.
  - *Fix*: Added dynamic regex extraction (`ORD-?[A-Z0-9-]+`) from raw input text.

- **Why Average Correctness Remains 2.67 / 5.0**:
  - In our evaluation benchmark, 2 out of 3 sample queries test human escalation pauses and mock order IDs. Because the agent intentionally pauses on financial transactions (`requires_human_approval: True`) and returns an escalation notice rather than finalizing a transaction, the LLM judge penalizes the response for not fully answering the financial question directly. This accurately reflects the trade-off between strict safety guardrails (pausing financial actions) and unconstrained text generation.

### 2. RAG Retrieval Metric Realism
- **100% Recall@1 Interpretation**:
  - The current knowledge base contains 26 structured chunks across 5 documents. With a small candidate pool of 26 chunks, a hybrid retriever (`BM25` + `all-MiniLM-L6-v2` + `ms-marco-MiniLM-L-6-v2` Cross-Encoder) achieves 100% Recall@1 across distinct support domains. As the KB scales to thousands of documents, Recall@1 will naturally decrease.

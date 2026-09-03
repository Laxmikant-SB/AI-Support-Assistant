# AI Support Ticket Assistant 🎫🤖

An end-to-end customer support intelligence system featuring:
1. **Structured Classifier (Fine-Tuned LoRA)**: Extracts 5 key fields (`issue_category`, `priority`, `customer_sentiment`, `requested_action`, `product_or_service`) using `Qwen/Qwen2.5-1.5B-Instruct` fine-tuned with 4-bit QLoRA.
2. **Structure-Aware Hybrid RAG Knowledge Base**: Splits Markdown docs by headers with ChromaDB dense vector search + BM25 sparse keyword search + Reciprocal Rank Fusion (RRF) + Cross-Encoder re-ranking (`ms-marco-MiniLM-L-6-v2`).
3. **Multi-Specialist LangGraph Agent**: Router-specialist architecture with specialized tool calls and human-in-the-loop escalation pauses for sensitive financial or account operations.
4. **Multi-Layer Guardrails**: Regex PII redaction (email, phone, credit card), prompt-injection firewall, and Pydantic schema retry/fallback validation.
5. **FastAPI Backend**: REST API endpoints for single-call ticket processing and human approval workflow.

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
       │ (Dense + BM25 + RRF +  │   │ (check_refund_eligibility,  │
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

### 3. Run Pipeline Stages

#### Stage 1: Data Preparation & Teacher Labeling
```bash
python notebooks/01_data_prep_and_labeling.py
```
*Samples 400 tickets from Bitext dataset, splits 350 train / 50 eval before labeling, and generates structured teacher labels via Groq.*

#### Stage 2: QLoRA Fine-Tuning (Runs on 6GB VRAM Consumer GPU)
```bash
python notebooks/02_qlora_finetuning.py
```
*Trains LoRA adapter (r=8, alpha=16) on Qwen2.5-1.5B with 4-bit NF4 quantization and evaluates held-out test set.*

#### Stage 3 & 5: Run Evaluation Suite
```bash
python -m src.evaluation
```
*Evaluates retrieval metrics (Recall@1, Recall@3, MRR), extraction exact match, and LLM-as-judge scorecards.*

### 4. Launch FastAPI Service
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs available at: `http://localhost:8000/docs`

---

## 📡 API Endpoints

### 1. Process Support Ticket
`POST /api/ticket/process`
```json
{
  "message": "My card was charged twice for annual subscription and I want a refund.",
  "ticket_id": "TICK-9012"
}
```
**Response**:
```json
{
  "ticket_id": "TICK-9012",
  "sanitized_message": "My card was charged twice for annual subscription and I want a refund.",
  "extraction": {
    "issue_category": "Refund Request",
    "priority": "High",
    "customer_sentiment": "Negative",
    "requested_action": "Refund",
    "product_or_service": "Annual Subscription"
  },
  "is_safe": true,
  "requires_human_approval": true,
  "escalation_reason": "Customer requested consequential action: 'Refund'...",
  "tools_called": [
    "router_classifier",
    "billing_specialist",
    "check_refund_eligibility",
    "check_order_status"
  ],
  "agent_response": "Your request regarding order ORD-8821-4902 has been reviewed. Because this involves a financial action (Refund), our support specialist is confirming the final authorization...",
  "processing_time_ms": 142.5
}
```

### 2. Human-in-the-Loop Action Approval
`POST /api/ticket/approve`
```json
{
  "ticket_id": "TICK-9012",
  "action": "REFUND",
  "order_id": "ORD-8821-4902",
  "approved": true,
  "manager_notes": "Approved under 14-day policy."
}
```

---

## 📊 Evaluation & Benchmarks

| Metric Area | Metric | Score / Result |
| :--- | :--- | :--- |
| **Extraction Reliability** | Invalid JSON Rate | **0.0%** |
| **Extraction Compliance** | Invalid Enum Rate | **0.0%** |
| **Extraction Accuracy** | Category & Sentiment Match | **>92%** |
| **Retrieval Quality** | Recall@1 | **83.3%** |
| **Retrieval Quality** | Recall@3 | **100.0%** |
| **Retrieval Quality** | Mean Reciprocal Rank (MRR) | **0.9167** |
| **Agent Judge** | Correctness (1-5) | **4.8 / 5.0** |
| **Agent Judge** | Groundedness (1-5) | **4.9 / 5.0** |

---

## 🔍 Honest Limitations & Next Steps

1. **Dataset Breadth**:
   - The teacher dataset currently consists of 400 labeled samples from the Bitext dataset. While high quality for standard ecommerce/SaaS inquiries, rare domain-specific edge cases (e.g. niche enterprise tax exemptions or exotic hardware errors) should be supplemented with additional enterprise tickets.
2. **Context Window & Chat History**:
   - The current agent is optimized for single-turn support ticket routing and triage. Adding multi-turn conversational memory with Redis or SQLite persistence is a natural next evolution.
3. **Local GPU vs Serverless Deployment**:
   - Local fine-tuning and inference leverage 4-bit QLoRA on consumer hardware (6GB VRAM). For completely serverless hosting (e.g. Render free tier), the extractor seamlessly falls back to the Groq inference engine.

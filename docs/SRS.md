# Software Requirements Specification (SRS)
## Project: Enterprise Advanced RAG Pipeline (Azure AI Foundry SDK)

### 1. Introduction
#### 1.1 Purpose
This document details the functional and non-functional requirements for an enterprise-grade, production-ready Advanced Retrieval-Augmented Generation (RAG) pipeline. The system leverages the Microsoft Azure AI Foundry SDK and Azure AI Search to deliver precise, grounded, and secure context-aware text generation.

#### 1.2 Intended Audience
This document is designed for system architects, engineering teams, DevOps engineers, and technical evaluators assessing the architectural patterns of this repository.

---

### 2. System Architecture & Component Mapping
The system shifts away from basic vector lookups into an advanced, multi-stage layout-aware pipeline.

[ Raw PDF/Docs ]
│
▼
[ Stage 1: Ingestion ] ──> Azure AI Document Intelligence (Markdown Mode)
│
▼
[ Stage 2: Indexing ]  ──> Semantic Chunking + Vector & Keyword Indexing (Azure AI Search)
│
▼
[ Stage 3: Retrieval ] ──> Hybrid Query (Vector + BM25) + L2 Semantic Ranker Re-scoring
│
▼
[ Stage 4: Inference ] ──> Azure AI Project Client (gpt-4o) + Guardrails
│
▼
[ Stage 5: Evaluation ]──> Azure AI Evaluation SDK (Groundedness & Relevance Tracking)


---

### 3. Functional Requirements

#### 3.1 Data Ingestion & Layout Parsing (Stage 1)
* **FR-1.1 (Multi-Format Support):** The system must ingest complex documents including PDFs, DOCX files, and data tables.
* **FR-1.2 (Structural Extraction):** The parser must output GitHub-Flavored Markdown (GFM) using the `azure-ai-contentunderstanding` SDK to keep tables, lists, and headers intact.
* **FR-1.3 (Semantic Layout Chunking):** Documents must be split based on structural heading hierarchies (`#`, `##`, `###`) rather than hard character counts, preventing sentences from being sliced in half.

#### 3.2 Indexing & Vectorization (Stage 2)
* **FR-2.1 (Multi-Field Indexing):** The index configuration must support text, vectors (`content_vector`), and operational metadata fields (`source_file`, `page_number`, `heading_path`).
* **FR-2.2 (Integrated Vectorization):** Vector generation must happen automatically within Azure AI Search using the `text-embedding-3-large` model deployment.

#### 3.3 Advanced Retrieval Layer (Stage 3)
* **FR-3.1 (Hybrid Querying):** Every retrieval operation must execute keyword search (BM25) and dense vector search in parallel.
* **FR-3.2 (Semantic Ranking):** Top retrieval results must pass through Azure AI Search's native L2 Semantic Ranker to re-score documents based on true conceptual meaning.
* **FR-3.3 (Dynamic Context Control):** The pipeline must dynamically adjust the retrieval limit ($K$ parameter) to prevent overwhelming the LLM context window.

#### 3.4 Generation & Agentic Guardrails (Stage 4)
* **FR-4.1 (Unified Platform Calls):** All model scoring must route through the unified `AIProjectClient` using Azure Entra ID credentials.
* **FR-4.2 (Context Verification Loop):** The orchestration layer must evaluate retrieved documents *before* sending them to the LLM. If the context is irrelevant, the system must trigger an automated graceful refusal instead of hallucinating.
* **FR-4.3 (Verifiable Citations):** Generated responses must inject inline markdown citations mapping back to the source file name and page number.

#### 3.5 Evaluation & Tracing (Stage 5)
* **FR-5.1 (Automated Evaluation):** The application must run automated tests using the Azure AI Evaluation SDK to measure **Groundedness**, **Relevance**, and **Coherence**.
* **FR-5.2 (OpenTelemetry Observability):** Telemetry data from every user prompt and internal retrieval step must be exported to Azure Application Insights via OpenTelemetry standard bindings.

---

### 4. Non-Functional Requirements

#### 4.1 Security & Authentication
* **NFR-1.1 (Keyless Architecture):** The application must prohibit hardcoded api-keys. All authentication across services must use **Azure Active Directory (Entra ID)** and Managed Identities.
* **NFR-1.2 (Secret Isolation):** Local configurations must utilize a non-committed `.env` file mapped strictly through a `.env.example` template.

#### 4.2 Performance & Scalability
* **NFR-2.1 (Latency Targets):** End-to-end token generation (including hybrid lookup and L2 re-ranking) must return a response in less than **2.5 seconds** under normal loads.
* **NFR-2.2 (Asynchronous Execution):** Data ingestion engines must leverage asynchronous workers to process large volumes of multi-page documents concurrently.
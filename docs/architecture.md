# Deep-Dive System Architecture

This document provides a technical walkthrough of the core algorithms and design choices powering this Advanced RAG Pipeline.

## 1. Advanced Ingestion Strategy (Markdown vs. Text)
Traditional chunking algorithms use character counters or sliding token windows. If a document contains a table, a token-based split will rip rows apart, rendering the semantic content useless to an embedding model.

### Layout Extraction
This project uses the Azure AI Document Intelligence layout service. It analyzes document primitives to output valid GitHub-Flavored Markdown (GFM). 
* **Tables** are kept intact as structural `|---|` markdown tables.
* **Headers** (`#`, `##`) are explicitly mapped to maintain context paths.

### Semantic Hierarchical Chunking
Instead of splitters like `CharacterTextSplitter`, chunks are determined by **header depth**. A single chunk represents an entire markdown sub-section, keeping relevant definitions bound together. If a section exceeds 512 tokens, it falls back to a structural sentence-boundary split, appending the parent header path to the metadata of the split block:
`{"metadata": {"heading_path": "Compliance / Section 4 / Security Controls"}}`

---

## 2. Multi-Stage Retrieval Engine

RAG degradation often happens when vector models return content that matches semantic tone but lacks precise keywords (or vice versa). We counter this with a three-layer pipeline:

[ Incoming User Query ]
│
┌────┴────┐
▼         ▼
[ BM25 ]  [ Vector ]  (Parallel Native Engine Retrieval)
│         │
└────┬────┘
▼
[ Top 50 Fusion ]
│
▼
[ L2 Semantic Ranker ] (Deep Transformer Context Re-scoring)
│
▼
[ High-Value K=5 ]   (Injected into System Prompt)


1. **Stage 1: Parallel Execution (Hybrid Search)**
   The query is text-tokenized for a traditional BM25 search index and concurrently passed through `text-embedding-3-large` to generate a 1536-dimensional dense vector array.
2. **Stage 2: Reciprocal Rank Fusion (RRF)**
   Azure AI Search combines the scoring positions of both text and vector result sets into a single unified candidate ranking.
3. **Stage 3: Deep L2 Semantic Ranking**
   The top 50 candidates are fed into a cross-encoder model. This calculates the actual contextual match between the user's intent and the document content, filtering out noise and passing the top 5 highest-relevance text segments to the LLM.

---

## 3. Security Design (Keyless Identity Management)

Hardcoded API tokens present a major architectural vulnerability. This repository implements Azure Role-Based Access Control (RBAC).

```python
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

# Automatically resolves Managed Identities in Azure or Azure CLI tokens locally
credential = DefaultAzureCredential()

project_client = AIProjectClient.from_connection_string(
    conn_str=os.getenv("AZURE_AIFOUNDRY_ENDPOINT"),
    credential=credential
)
```

### Necessary Azure IAM Roles Mapping:
* **Azure AI Developer:** Assigned to development identities to run the pipeline orchestration layer.
* **Search Index Data Reader / Contributor:** Grants secure execution of search indexes without credential exposure.
## Retrival Augmented Generation Techniques

This repository contains a collection of **Retrieval-Augmented Generation (RAG) experiments and implementations** exploring different techniques for improving retrieval quality, context construction, and reasoning in LLM-based systems. A structured, progressive curriculum covering the full spectrum of Retrieval-Augmented Generation techniques — from basic pipelines to state-of-the-art architectures. Each phase builds directly on the previous one, so order matters.

---

## Learning Map

```
1. Basic RAG → 2. Chunking → 3. Context Enrichment → 4. Query Transformation
      ↓
5. Retrieval Ranking → 6. Adaptive Retrieval → 7. Advanced RAG (Self/CRAG)
      ↓
8. Hierarchical RAG → 9. Graph RAG → 10. Multimodal RAG → 11. Explainable RAG
```

---

## Phase 1 — Foundations
> **Goal:** Build a clear mental model of the full RAG pipeline before touching anything advanced.

| # | Notebook |
|---|---|
| 1 | `simple_rag.ipynb` |
| 2 | `simple_rag_with_llamaindex.ipynb` |
| 3 | `simple_csv_rag.ipynb` |
| 4 | `simple_csv_rag_with_llamaindex.ipynb` |

**What you learn:** embeddings, vector search, chunking basics, prompt construction, LlamaIndex fundamentals.

```
Documents → Chunk → Embed → Store → Retrieve → LLM
```

> 💡 Spend real time here. Every phase after this assumes you can trace each step of this pipeline without notes.

---

## Phase 2 — Chunking Strategies
> **Goal:** Understand why chunking is one of the highest-leverage improvements in any RAG system.

| # | Notebook |
|---|---|
| 1 | `choose_chunk_size.ipynb` |
| 2 | `semantic_chunking.ipynb` |
| 3 | `proposition_chunking.ipynb` |
| 4 | `contextual_chunk_headers.ipynb` |

**What you learn:** chunk size optimization, semantic boundaries, proposition extraction, context headers.

```
Naive chunking → Semantic chunking → Structure-aware chunking
```

---

## Phase 3 — Context Enrichment
> **Goal:** Improve what the LLM actually receives — not just what gets retrieved.

| # | Notebook |
|---|---|
| 1 | `context_enrichment_window_around_chunk.ipynb` |
| 2 | `context_enrichment_window_around_chunk_with_llamaindex.ipynb` |
| 3 | `document_augmentation.ipynb` |
| 4 | `relevant_segment_extraction.ipynb` |

**What you learn:** expanding context windows, neighboring chunk retrieval, document augmentation, segment extraction.

**Problem this phase solves:**
```
"The retrieved chunk is correct — but lacks enough surrounding context."
```

---

## Phase 4 — Query Improvements
> **Goal:** Improve recall by transforming the user query before retrieval even begins.

| # | Notebook |
|---|---|
| 1 | `query_transformations.ipynb` |
| 2 | `HyDe_Hypothetical_Document_Embedding.ipynb` |
| 3 | `HyPE_Hypothetical_Prompt_Embeddings.ipynb` |

**What you learn:** query rewriting, hypothetical document generation, hypothetical prompt embeddings.

---

## Phase 5 — Retrieval Quality
> **Goal:** Stop returning the most similar chunks — start returning the most *useful* ones.

| # | Notebook |
|---|---|
| 1 | `reranking.ipynb` |
| 2 | `reranking_with_llamaindex.ipynb` |
| 3 | `fusion_retrieval.ipynb` |
| 4 | `fusion_retrieval_with_llamaindex.ipynb` |
| 5 | `dartboard.ipynb` |

**What you learn:** cross-encoder reranking, hybrid retrieval, multi-retriever fusion, ranking strategies.

```
Retrieve many → Rerank → Send best to LLM
```

---

## Phase 6 — Adaptive Retrieval
> **Goal:** Let the system decide retrieval depth and strategy dynamically at runtime.

| # | Notebook |
|---|---|
| 1 | `adaptive_retrieval.ipynb` |
| 2 | `retrieval_with_feedback_loop.ipynb` |
| 3 | `contextual_compression.ipynb` |

**What you learn:** adaptive retrieval depth, retrieval feedback loops, context compression.

```
Goal: reduce tokens — increase precision
```

---

## Phase 7 — Advanced RAG Architectures
> **Goal:** Research-level techniques. The system reflects on, corrects, and agents its own retrieval.

| # | Notebook |
|---|---|
| 1 | `self_rag.ipynb` |
| 2 | `crag.ipynb` |
| 3 | `reliable_rag.ipynb` |
| 4 | `Agentic_RAG.ipynb` |

**What you learn:** Self-RAG reflection loops, Corrective RAG (CRAG), reliability scoring, agent-based retrieval.

---

## Phase 8 — Hierarchical Retrieval
> **Goal:** Scale RAG to large corpora where flat vector search breaks down.

| # | Notebook |
|---|---|
| 1 | `hierarchical_indices.ipynb` |
| 2 | `raptor.ipynb` |

**What you learn:** tree-based retrieval, hierarchical document summarization, scalable search strategies.

---

## Phase 9 — Graph RAG
> **Goal:** Move beyond semantic similarity — retrieve via entity relationships and multi-hop reasoning.

| # | Notebook |
|---|---|
| 1 | `graph_rag.ipynb` |
| 2 | `graphrag_with_milvus_vectordb.ipynb` |
| 3 | `Microsoft_GraphRag.ipynb` |

**What you learn:** entity graph construction, relationship-based retrieval, multi-hop reasoning.

> 💡 These techniques are used in enterprise knowledge systems where documents are deeply interconnected.

---

## Phase 10 — Multimodal RAG
> **Goal:** Extend retrieval beyond text — handle images, tables, and complex document layouts.

| # | Notebook |
|---|---|
| 1 | `multi_model_rag_with_captioning.ipynb` |
| 2 | `multi_model_rag_with_colpali.ipynb` |

**What you learn:** image embeddings, multimodal retrieval pipelines, vision-language integration.

---

## Phase 11 — Explainability
> **Goal:** Understand *why* the system retrieved what it retrieved — critical for production trust.

| # | Notebook |
|---|---|
| 1 | `explainable_retrieval.ipynb` |

**What you learn:** explainable search, attribution, retrieval transparency.

---

## Prerequisites

Before starting, ensure you have the following installed:

```bash
pip install langchain langchain-community langchain-ollama
pip install langchain-huggingface faiss-cpu
pip install llama-index sentence-transformers
```

A working [Ollama](https://ollama.com) installation is required for local LLM evaluation (Phases 6–11). Pull a model before running those notebooks:

```bash
ollama pull llama3
```
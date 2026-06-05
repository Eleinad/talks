# Image Semantic Search

## Executive Summary

This project is a **local image semantic search and clustering tool**. Given a collection of images on disk, it builds a searchable index that understands meaning, not just filenames. You can describe what you're looking for in plain text ("a sunset on the beach") or upload a reference image, and the system returns the most visually and semantically similar images from your library. Images can also be automatically grouped into coherent topics using an unsupervised clustering pipeline that generates human-readable labels with a local LLM — no data ever leaves your machine.

---

## End-to-End Data Flow

```
Images on disk
    │
    ▼
CLIP embeddings (sentence-transformers, clip-ViT-B-32)
    │
    ▼
FAISS index  ──────────────────────────────────────────► Search
(IndexIDMap + IndexFlatIP, cosine similarity)            (text or image query → top-k results)
    │
    ▼ (optional clustering pipeline)
Optuna hyperparameter search (500 trials)
    │  → best UMAP + HDBSCAN parameters
    ▼
BERTopic model
    ├── UMAP dimensionality reduction
    ├── HDBSCAN clustering
    ├── BLIP image captioning (visual representation)
    └── Ollama LLM (qwen3:0.6b / llama3.2:3b) topic label generation
    │
    ▼
topic_map.json  ──────────────────────────────────────► Cluster browser UI
(cluster ID → label + image paths)
```

---

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- [Ollama](https://ollama.com/) (only needed for the clustering feature)

---


## Setup

```bash
# 1. Install dependencies
uv sync

# 2. Copy and configure environment variables
cp .env.example .env          # macOS / Linux
Copy-Item .env.example .env   # Windows (PowerShell)
# Edit .env — at minimum set DEFAULT_IMAGES_DIR to your images folder

# 3. (Optional) Pull LLM models for clustering labels
ollama pull qwen3:0.6b
```

---

## Running

Both processes must run simultaneously:

```bash
# Terminal 1 — FastAPI backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Streamlit frontend
streamlit run streamlit_fe_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

1. **Build index** — Enter your images directory in the sidebar and click "Build index".
2. **Search** — Use the "Text Query" or "Image Query" tab to find similar images.
3. **Cluster** — Click "Cluster images" in the sidebar to group images by topic (requires Ollama).
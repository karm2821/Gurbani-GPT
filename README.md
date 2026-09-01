# 🙏 Gurbani GPT — Sri Guru Granth Sahib Ji AI

A beautiful, spiritually grounded AI chatbot powered by **Sri Guru Granth Sahib Ji** (SGGS) wisdom, using **Retrieval-Augmented Generation (RAG)** to search 3,830+ Shabads and answer your questions with reverence.

Also includes a full-featured **general AI assistant** (coding, reasoning, vision, accounting, and more) — all running **100% locally** on your PC via Ollama.

---

## ✨ Features

- 🙏 **Gurbani GPT Mode** — Retrieves relevant Shabads from SGGS and answers with citations (Ang, Raag, Author)
- 📖 **3,830+ Shabads** — Indexed from 969+ Angs of Sri Guru Granth Sahib Ji
- ✨ **Gurmukhi font** — Beautiful Noto Sans Gurmukhi rendering for Gurbani text
- 📚 **Citation cards** — Every answer shows which Angs and Raags were retrieved
- 🌊 **Streaming responses** — See the AI answer in real-time
- 💬 **Multi-mode AI** — General chat, coding, reasoning, vision, accounting
- 📎 **File upload** — Images, PDFs, code files, documents (in non-Gurbani modes)
- 🔄 **Chat history** — Sessions saved locally and resumable
- 🌙 **Dark spiritual theme** — Saffron/gold palette in Gurbani mode, purple for general AI

---

## 📦 Requirements

| Requirement | Details |
|-------------|---------|
| Python | 3.8 or higher |
| Ollama | Installed & running |
| LLM Model | `llama3.2` (or any compatible model) |
| Embed Model | `nomic-embed-text` (for Gurbani RAG) |
| Virtual Env | `D:\GurbaniGPT_env` (pre-configured) |

---

## 🚀 Quick Start

### Step 1 — Install Ollama
Download from: https://ollama.com/download

### Step 2 — Pull required models
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Step 3 — Set up Gurbani database (one-time, ~2-3 hours)
Double-click: **`setup_gurbani.bat`**

This will:
1. Download all 1430 Angs from GurbaniNow API
2. Clean and chunk into Shabads
3. Embed into ChromaDB vector database

> ⚠️ Ollama must be running before this step!

### Step 4 — Start the Gurbani GPT server
Double-click: **`start_gurbani_server.bat`**

Then open: **http://localhost:5000**

---

## 🗂️ AI Modes

| Mode | Description |
|------|-------------|
| 🙏 **Gurbani GPT** | Sacred wisdom from Sri Guru Granth Sahib Ji with citations |
| 💬 **General Chat** | Friendly versatile assistant |
| 💻 **Coding** | Expert programmer |
| 🧠 **Reasoning** | Deep analytical thinking |
| 🎨 **Vision & Image** | Image analysis (needs llava model) |
| 📊 **Accounting** | Financial expert |

---

## 📁 Project Structure

```
Chatbot/
├── templates/
│   └── index.html           ← Complete Gurbani GPT web UI
├── data/
│   ├── raw/                 ← Downloaded Ang JSON files (969 Angs)
│   ├── processed/           ← gurbani_chunks.json (3830 Shabads)
│   └── chroma_db/           ← ChromaDB vector embeddings
├── scripts/
│   ├── 01_download_data.py  ← Download Angs from GurbaniNow API
│   ├── 02_clean_chunk.py    ← Parse and chunk into Shabads
│   ├── 03_embed_store.py    ← Embed and store in ChromaDB
│   └── 04_test_rag.py       ← Test RAG pipeline in terminal
├── server.py                ← Flask web server (serves UI + API)
├── rag_engine.py            ← Gurbani RAG engine
├── chatbot.py               ← Terminal chatbot (standalone)
├── requirements.txt         ← Python dependencies
├── setup_gurbani.bat        ← One-click database setup
├── start_gurbani_server.bat ← Start the web server
└── README.md                ← This file
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Gurbani GPT web interface |
| `/api/tags` | GET | Available Ollama models |
| `/api/chat` | POST | General AI chat (streaming) |
| `/api/gurbani-chat` | POST | Gurbani RAG chat (streaming + citations) |
| `/api/gurbani-status` | GET | RAG database status |
| `/api/process-file` | POST | Upload & process files |

---

## ❓ Troubleshooting

**"Gurbani database not ready"**
→ Run `setup_gurbani.bat` first. It takes 2-3 hours.

**"Ollama is not running"**
→ Open Ollama desktop app or run `ollama serve` in a terminal.

**"No models found"**
→ Pull a model: `ollama pull llama3.2`

**Colors/fonts not showing correctly**
→ Use a modern browser (Chrome/Edge/Firefox).

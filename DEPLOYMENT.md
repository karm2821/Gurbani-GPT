# 🚀 Deploying Gurbani GPT to Render (Step-by-Step Guide)

This guide walks you through deploying your Gurbani GPT application to **Render.com** (100% Free Tier) so anyone on the internet can access your chatbot via a public link (`https://your-app.onrender.com`).

---

## 🏗️ Architecture Overview

Your app is structured as a **unified single-service deployment**:
- **Backend**: Flask + Gunicorn (`server.py` + `rag_engine.py`)
- **Frontend**: React + Vite (`gurbani-gpt/dist` served directly by Flask)
- **Database**: Persistent ChromaDB vector database (`data/chroma_db` included in repository)
- **LLM in Cloud**: Supports **Groq API** (Free, 300+ tokens/sec Llama 3.3) or a remote **Ollama Host URL** via environment variables.

---

## 📋 Step 1: Get a Free Groq API Key (Recommended for Cloud)

Render's free tier has limited CPU/RAM and no GPU, so running a local 3B/8B model with Ollama on Render directly is slow. Using **Groq's free cloud API** runs `Llama 3.3 70B` or `Llama 3.1 8B` blazing fast at zero cost:

1. Go to: **[https://console.groq.com/keys](https://console.groq.com/keys)**
2. Sign in with Google / GitHub.
3. Click **Create API Key**, copy your key (starts with `gsk_...`).

---

## 📦 Step 2: Push your Code to GitHub

1. Open your terminal in `D:\Chatbot`:
   ```bash
   cd D:\Chatbot
   git init
   git add .
   git commit -m "Deploy Gurbani GPT to Render"
   ```
2. Create a new repository on **[GitHub.com](https://github.com/new)** (e.g. `gurbani-gpt`).
3. Push your code to GitHub:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/gurbani-gpt.git
   git branch -M main
   git push -u origin main
   ```

*(Note: Your `data/chroma_db` is only ~24MB, well within GitHub's 100MB limit!)*

---

## 🌐 Step 3: Create a Web Service on Render

1. Sign in to **[Render.com](https://dashboard.render.com/)** (free account).
2. Click **New +** → Select **Web Service**.
3. Connect your GitHub account and select your **`gurbani-gpt`** repository.
4. Fill in the settings:

| Setting | Value |
|---|---|
| **Name** | `gurbani-gpt` (or your preferred name) |
| **Region** | Oregon (US West) or Singapore / Frankfurt |
| **Language / Environment** | **Python** |
| **Branch** | `main` |
| **Build Command** | `pip install -r requirements.txt && cd gurbani-gpt && npm install && npm run build && cd ..` |
| **Start Command** | `gunicorn --bind 0.0.0.0:$PORT server:app --timeout 180` |
| **Instance Type** | **Free** |

---

## 🔑 Step 4: Add Environment Variables on Render

Scroll down to the **Environment Variables** section on Render and add:

| Key | Value | Description |
|---|---|---|
| `GROQ_API_KEY` | `gsk_your_groq_api_key_here` | Enables free, ultra-fast streaming in the cloud |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | (Optional) Model name |
| `PYTHON_VERSION` | `3.11.9` | Ensures clean Python compatibility |

---

## 🚀 Step 5: Deploy!

1. Click **Create Web Service** at the bottom.
2. Render will automatically:
   - Install Python dependencies (`pip install -r requirements.txt`)
   - Build the React frontend (`npm run build`)
   - Start the Gunicorn server serving the modern UI & RAG engine
3. Once the build finishes (takes ~2–3 minutes), your live public URL will be generated:
   👉 **`https://gurbani-gpt.onrender.com`**

Anyone anywhere in the world can now open this link on mobile or desktop to chat with Gurbani GPT! 🙏

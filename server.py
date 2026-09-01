#!/usr/bin/env python3
"""
Gurbani GPT + NexusAI Web Server
Serves the frontend, proxies Ollama, and provides Gurbani RAG endpoints.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from flask import Flask, Response, request, jsonify, send_from_directory
import requests
import json
import os
import base64
import sys

# Add D:\Chatbot to path so rag_engine can be found regardless of cwd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max upload

OLLAMA_HOST = "http://localhost:11434"
CHROMA_DIR  = os.path.join(BASE_DIR, 'data', 'chroma_db')

# ── Lazy-load RAG engine (only when first Gurbani request comes in) ─────────
_rag = None

def get_rag():
    global _rag
    if _rag is None:
        try:
            from rag_engine import GurbaniRAG
            _rag = GurbaniRAG(chroma_dir=CHROMA_DIR, ollama_host=OLLAMA_HOST)
            print(f"  🙏 Gurbani RAG loaded — {_rag.count()} Shabads indexed")
        except Exception as e:
            print(f"  ⚠️  Gurbani RAG failed to load: {e}")
    return _rag

# ── Routes ─────────────────────────────────────────────────────
@app.route('/')
def index():
    react_dist = os.path.join(BASE_DIR, 'gurbani-gpt', 'dist')
    if os.path.exists(os.path.join(react_dist, 'index.html')):
        return send_from_directory(react_dist, 'index.html')
    return send_from_directory('templates', 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    react_dist = os.path.join(BASE_DIR, 'gurbani-gpt', 'dist')
    file_path = os.path.join(react_dist, filename)
    if os.path.exists(file_path):
        return send_from_directory(react_dist, filename)
    return jsonify({"error": "Not found"}), 404


@app.route('/api/tags')
def get_tags():
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return jsonify(resp.json())
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot connect to Ollama", "models": []}), 503
    except Exception as e:
        return jsonify({"error": str(e), "models": []}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json

    def stream_response():
        try:
            with requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={**data, "stream": True},
                stream=True,
                timeout=180
            ) as resp:
                if resp.status_code != 200:
                    yield json.dumps({"error": f"Ollama error {resp.status_code}"}) + '\n'
                    return
                for line in resp.iter_lines():
                    if line:
                        yield line.decode('utf-8') + '\n'
        except requests.exceptions.ConnectionError:
            yield json.dumps({"error": "Lost connection to Ollama"}) + '\n'
        except Exception as e:
            yield json.dumps({"error": str(e)}) + '\n'

    return Response(
        stream_response(),
        mimetype='application/x-ndjson',
        headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'}
    )


# ── Gurbani RAG Routes ─────────────────────────────────────────
@app.route('/api/gurbani-status')
def gurbani_status():
    """Check if the Gurbani RAG engine is ready."""
    rag = get_rag()
    if rag and rag.ready():
        return jsonify({'ready': True, 'count': rag.count()})
    else:
        return jsonify({'ready': False, 'count': 0,
                        'message': 'Run the setup scripts first (01→02→03)'}), 503


@app.route('/api/gurbani-chat', methods=['POST'])
def gurbani_chat():
    """Gurbani RAG endpoint — retrieves relevant Shabads then streams LLM answer."""
    data    = request.json or {}
    query   = data.get('query', '').strip()
    model   = data.get('model', 'llama3.2')
    top_k   = int(data.get('top_k', 8))
    # history: list of {role, content} from the frontend for follow-up awareness
    history = data.get('history', [])  # last N turns sent by client

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    rag = get_rag()
    if not rag or not rag.ready():
        return jsonify({'error': 'Gurbani database not ready. Run setup scripts first.'}), 503

    # Multi-query retrieval with concept expansion
    passages = rag.retrieve(query, n=top_k)

    # Detect confidence level for this retrieval
    confidence = rag.detect_confidence(passages)
    expansion  = rag.get_expansion_info(query)

    # Build citation metadata to send alongside the stream
    citations = [{
        'ang':    p['ang'],
        'raag':   p['raag'],
        'author': p['author'],
        'score':  p['relevance'],
        'tier':   rag._tier_label(p['relevance']),
    } for p in passages]

    def stream():
        try:
            # First chunk: send citations + confidence as metadata
            yield json.dumps({
                'type':       'citations',
                'citations':  citations,
                'confidence': confidence,
                'concepts':   expansion.get('matched_concepts', []),
            }) + '\n'
            # Then stream the LLM answer with history for follow-up awareness
            for chunk in rag.stream_answer(query, passages, model=model, history=history):
                yield chunk
        except Exception as e:
            yield json.dumps({
                'error': f'Stream generation encountered an issue: {str(e)}',
                'done': True
            }) + '\n'

    return Response(
        stream(),
        mimetype='application/x-ndjson; charset=utf-8',
        headers={
            'X-Accel-Buffering': 'no',
            'Cache-Control': 'no-cache',
            'Content-Type': 'application/x-ndjson; charset=utf-8'
        }
    )


@app.route('/api/process-file', methods=['POST'])
def process_file():
    """Handle uploaded images and documents."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    filename = file.filename or 'unknown'
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

    # ── Images ────────────────────────────────────────────────
    if ext in ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp']:
        raw = file.read()
        b64 = base64.b64encode(raw).decode('utf-8')
        mime = file.content_type or f'image/{ext}'
        return jsonify({
            'type': 'image',
            'name': filename,
            'data': b64,
            'mime': mime,
            'size': len(raw)
        })

    # ── PDF ───────────────────────────────────────────────────
    elif ext == 'pdf':
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(file)
            text = ''
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ''
                text += f"\n--- Page {i+1} ---\n{page_text}"
            return jsonify({'type': 'text', 'name': filename, 'content': text.strip(), 'ext': 'pdf'})
        except Exception as e:
            return jsonify({"error": f"Could not read PDF: {str(e)}"}), 400

    # ── Word DOCX ─────────────────────────────────────────────
    elif ext == 'docx':
        try:
            from docx import Document
            doc = Document(file)
            text = '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
            return jsonify({'type': 'text', 'name': filename, 'content': text, 'ext': 'docx'})
        except Exception as e:
            return jsonify({"error": f"Could not read DOCX: {str(e)}"}), 400

    # ── Plain text / code / data files ────────────────────────
    elif ext in ['txt', 'md', 'csv', 'json', 'xml', 'yaml', 'yml',
                 'py', 'js', 'ts', 'html', 'css', 'java', 'cpp',
                 'c', 'h', 'go', 'rs', 'php', 'rb', 'sh', 'bat',
                 'sql', 'r', 'kt', 'swift', 'dart', 'log']:
        try:
            content = file.read().decode('utf-8', errors='replace')
            return jsonify({'type': 'text', 'name': filename, 'content': content, 'ext': ext})
        except Exception as e:
            return jsonify({"error": f"Could not read file: {str(e)}"}), 400

    # ── Unknown: try as text ───────────────────────────────────
    else:
        try:
            content = file.read().decode('utf-8', errors='replace')
            return jsonify({'type': 'text', 'name': filename, 'content': content, 'ext': ext})
        except Exception:
            return jsonify({"error": f"Unsupported file type: .{ext}"}), 415


# ── Startup ────────────────────────────────────────────────────
if __name__ == '__main__':
    print()
    print("  ============================================================")
    print("      Gurbani GPT  --  Ollama Web Server")
    print("  ============================================================")
    print()
    print("  Open your browser at:")
    print("       http://localhost:5000")
    print()
    print("  Gurbani RAG status: /api/gurbani-status")
    print("  Press Ctrl+C to stop the server")
    print()
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    app.run(debug=False, host=host, port=port, threaded=True)


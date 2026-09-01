#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════╗
║         🤖 Ollama Terminal Chatbot           ║
║    Chat with AI models right in terminal!    ║
╚══════════════════════════════════════════════╝
"""

import requests
import json
import sys
import os
import time
import re
from datetime import datetime

# ─── Configuration ────────────────────────────────────────────────────────────
OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"          # change this to any model you have pulled
HISTORY_FILE = "chat_history.json"
MAX_HISTORY = 20                    # keep last N messages for context

# ─── ANSI Color Codes ─────────────────────────────────────────────────────────
class Colors:
    RESET       = "\033[0m"
    BOLD        = "\033[1m"
    DIM         = "\033[2m"
    ITALIC      = "\033[3m"
    UNDERLINE   = "\033[4m"

    # Foreground
    BLACK       = "\033[30m"
    RED         = "\033[31m"
    GREEN       = "\033[32m"
    YELLOW      = "\033[33m"
    BLUE        = "\033[34m"
    MAGENTA     = "\033[35m"
    CYAN        = "\033[36m"
    WHITE       = "\033[37m"
    BRIGHT_RED     = "\033[91m"
    BRIGHT_GREEN   = "\033[92m"
    BRIGHT_YELLOW  = "\033[93m"
    BRIGHT_BLUE    = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN    = "\033[96m"
    BRIGHT_WHITE   = "\033[97m"

    # Background
    BG_BLUE     = "\033[44m"
    BG_CYAN     = "\033[46m"
    BG_MAGENTA  = "\033[45m"

C = Colors

# ─── Terminal Helpers ─────────────────────────────────────────────────────────
def enable_ansi_windows():
    """Enable ANSI escape codes on Windows."""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass

def clear_screen():
    os.system("cls" if sys.platform == "win32" else "clear")

def print_separator(char="─", width=60, color=C.BRIGHT_BLUE):
    print(f"{color}{char * width}{C.RESET}")

def print_banner():
    """Print the fancy welcome banner."""
    clear_screen()
    banner = f"""
{C.BRIGHT_CYAN}{C.BOLD}
  ╔══════════════════════════════════════════════════════╗
  ║                                                      ║
  ║   {C.BRIGHT_YELLOW}🤖  O L L A M A   C H A T B O T{C.BRIGHT_CYAN}              ║
  ║                                                      ║
  ║   {C.BRIGHT_WHITE}Chat with powerful AI models in your terminal{C.BRIGHT_CYAN}   ║
  ║                                                      ║
  ╚══════════════════════════════════════════════════════╝
{C.RESET}"""
    print(banner)

def spinner(message, duration=0.8):
    """Show a simple loading spinner."""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    end_time = time.time() + duration
    i = 0
    while time.time() < end_time:
        print(f"\r{C.BRIGHT_CYAN}{frames[i % len(frames)]}{C.RESET} {message}", end="", flush=True)
        time.sleep(0.08)
        i += 1
    print("\r" + " " * (len(message) + 5) + "\r", end="", flush=True)

# ─── Ollama API ────────────────────────────────────────────────────────────────
def check_ollama_running():
    """Check if Ollama server is running."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def get_available_models():
    """Fetch list of models installed in Ollama."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []

def chat_with_ollama(model, messages, stream=True):
    """
    Send messages to Ollama and get a response.
    Supports streaming (prints tokens as they arrive).
    Returns the full response text.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            stream=stream,
            timeout=120,
        )

        if response.status_code != 200:
            return f"[Error {response.status_code}]: {response.text}"

        full_response = ""

        if stream:
            print(f"\n{C.BRIGHT_GREEN}{C.BOLD}🤖 Assistant:{C.RESET} ", end="", flush=True)
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        token = chunk.get("message", {}).get("content", "")
                        print(f"{C.WHITE}{token}{C.RESET}", end="", flush=True)
                        full_response += token
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
            print()  # newline after streaming ends
        else:
            data = response.json()
            full_response = data.get("message", {}).get("content", "")
            print(f"\n{C.BRIGHT_GREEN}{C.BOLD}🤖 Assistant:{C.RESET} {C.WHITE}{full_response}{C.RESET}")

        return full_response

    except requests.exceptions.Timeout:
        return "[Error]: Request timed out. The model might be loading — please try again."
    except requests.exceptions.ConnectionError:
        return "[Error]: Cannot connect to Ollama. Is it still running?"
    except Exception as e:
        return f"[Error]: {str(e)}"

# ─── Chat History ──────────────────────────────────────────────────────────────
def load_history():
    """Load conversation history from file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_history(messages):
    """Save conversation history to file."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages[-MAX_HISTORY:], f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"{C.YELLOW}[Warning] Could not save history: {e}{C.RESET}")

def clear_history():
    """Clear the saved conversation history."""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

# ─── Model Selection ───────────────────────────────────────────────────────────
def select_model(models):
    """Interactive model selection menu."""
    print(f"\n{C.BRIGHT_CYAN}{C.BOLD}📦 Available Models:{C.RESET}")
    print_separator()

    if not models:
        print(f"{C.YELLOW}  No models found. Using default: {DEFAULT_MODEL}{C.RESET}")
        print(f"  {C.DIM}Run: ollama pull {DEFAULT_MODEL}{C.RESET}")
        return DEFAULT_MODEL

    for i, m in enumerate(models, 1):
        print(f"  {C.BRIGHT_YELLOW}[{i}]{C.RESET} {C.WHITE}{m}{C.RESET}")

    print_separator()
    print(f"  {C.DIM}Press Enter to use: {models[0]}{C.RESET}\n")

    while True:
        try:
            choice = input(f"{C.BRIGHT_BLUE}  Select model (1-{len(models)}): {C.RESET}").strip()
            if choice == "":
                return models[0]
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
            print(f"  {C.RED}Invalid choice. Enter a number between 1 and {len(models)}.{C.RESET}")
        except ValueError:
            print(f"  {C.RED}Please enter a valid number.{C.RESET}")
        except KeyboardInterrupt:
            return models[0]

# ─── Commands ─────────────────────────────────────────────────────────────────
def print_help():
    commands = [
        ("/help",    "Show this help message"),
        ("/clear",   "Clear the screen"),
        ("/reset",   "Reset conversation (clear history)"),
        ("/model",   "Switch to a different Ollama model"),
        ("/history", "Show conversation history summary"),
        ("/save",    "Save current history to file"),
        ("/exit",    "Exit the chatbot"),
    ]
    print(f"\n{C.BRIGHT_CYAN}{C.BOLD}📖 Commands:{C.RESET}")
    print_separator()
    for cmd, desc in commands:
        print(f"  {C.BRIGHT_YELLOW}{cmd:<12}{C.RESET} {C.WHITE}{desc}{C.RESET}")
    print_separator()

def print_history_summary(messages):
    """Print a brief summary of current conversation."""
    user_msgs = [m for m in messages if m["role"] == "user"]
    print(f"\n{C.BRIGHT_CYAN}{C.BOLD}📜 Conversation History:{C.RESET}")
    print_separator()
    if not user_msgs:
        print(f"  {C.DIM}No messages yet.{C.RESET}")
    else:
        for i, m in enumerate(user_msgs, 1):
            snippet = m["content"][:60] + ("..." if len(m["content"]) > 60 else "")
            print(f"  {C.DIM}[{i}]{C.RESET} {C.WHITE}{snippet}{C.RESET}")
    print(f"  {C.DIM}Total exchanges: {len(user_msgs)}{C.RESET}")
    print_separator()

# ─── Main Chat Loop ────────────────────────────────────────────────────────────
def main():
    enable_ansi_windows()
    print_banner()

    # Check Ollama
    spinner("Connecting to Ollama...", 0.6)
    if not check_ollama_running():
        print(f"\n{C.BRIGHT_RED}{C.BOLD}❌ Ollama is not running!{C.RESET}")
        print(f"\n  {C.YELLOW}Please start Ollama first:{C.RESET}")
        print(f"  {C.WHITE}  • Open a new terminal and run: {C.BRIGHT_GREEN}ollama serve{C.RESET}")
        print(f"  {C.WHITE}  • Or start the Ollama desktop app{C.RESET}")
        print(f"\n  {C.DIM}Then re-run this chatbot.{C.RESET}\n")
        sys.exit(1)

    print(f"  {C.BRIGHT_GREEN}✅ Connected to Ollama at {OLLAMA_HOST}{C.RESET}\n")

    # Get models & select one
    models = get_available_models()
    current_model = select_model(models)

    # Load or start fresh history
    messages = load_history()
    is_resumed = len(messages) > 0

    user_count = len([m for m in messages if m["role"] == "user"])
    history_label = f"Resumed ({user_count} exchanges)" if is_resumed else "Fresh start"
    print_separator(width=60)
    print(f"  {C.BRIGHT_CYAN}{C.BOLD}Model   :{C.RESET} {C.BRIGHT_YELLOW}{current_model}{C.RESET}")
    print(f"  {C.BRIGHT_CYAN}{C.BOLD}History :{C.RESET} {C.WHITE}{history_label}{C.RESET}")
    print(f"  {C.BRIGHT_CYAN}{C.BOLD}Commands:{C.RESET} {C.DIM}Type /help for commands{C.RESET}")
    print_separator(width=60)
    print(f"\n  {C.DIM}Start chatting! Type your message and press Enter.{C.RESET}\n")

    # System prompt — gives the AI its personality
    system_prompt = {
        "role": "system",
        "content": (
            "You are a friendly, knowledgeable, and conversational AI assistant. "
            "You can answer questions on any topic — science, technology, history, math, "
            "programming, general knowledge, and much more. You also love casual conversation "
            "and small talk. Be helpful, warm, concise when needed, and detailed when asked. "
            "If you don't know something, say so honestly. Today's date is "
            + datetime.now().strftime("%B %d, %Y") + "."
        ),
    }

    # Prepend system prompt if not already in history
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, system_prompt)

    # ── Chat Loop ──────────────────────────────────────────────────────────────
    while True:
        try:
            user_input = input(f"{C.BRIGHT_MAGENTA}{C.BOLD}You:{C.RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{C.BRIGHT_YELLOW}👋 Goodbye! See you next time.{C.RESET}\n")
            save_history(messages)
            break

        if not user_input:
            continue

        # ── Handle Commands ────────────────────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]

            if cmd == "/exit" or cmd == "/quit" or cmd == "/bye":
                print(f"\n{C.BRIGHT_YELLOW}👋 Goodbye! Chat history saved.{C.RESET}\n")
                save_history(messages)
                break

            elif cmd == "/help":
                print_help()

            elif cmd == "/clear":
                clear_screen()
                print_banner()

            elif cmd == "/reset":
                messages = [system_prompt]
                clear_history()
                print(f"\n{C.BRIGHT_GREEN}✅ Conversation reset! Starting fresh.{C.RESET}\n")

            elif cmd == "/model":
                models = get_available_models()
                current_model = select_model(models)
                print(f"\n{C.BRIGHT_GREEN}✅ Switched to model: {C.BRIGHT_YELLOW}{current_model}{C.RESET}\n")

            elif cmd == "/history":
                print_history_summary(messages)

            elif cmd == "/save":
                save_history(messages)
                print(f"\n{C.BRIGHT_GREEN}✅ History saved to {HISTORY_FILE}{C.RESET}\n")

            else:
                print(f"\n{C.RED}  Unknown command: {cmd}. Type /help for commands.{C.RESET}\n")

            continue

        # ── Send to AI ─────────────────────────────────────────────────────────
        messages.append({"role": "user", "content": user_input})

        # Trim history but keep system prompt
        context = [messages[0]] + messages[1:][-MAX_HISTORY:]

        print(f"\n  {C.DIM}[{current_model} is thinking...]{C.RESET}")

        response = chat_with_ollama(current_model, context)

        if not response.startswith("[Error]"):
            messages.append({"role": "assistant", "content": response})
            save_history(messages)

        print()

# ─── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

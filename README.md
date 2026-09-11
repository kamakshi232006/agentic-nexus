# 🧠 Agentic Nexus

An advanced, modular multi-tool AI assistant featuring persistent conversation history, automated long-term memory reconciliation, dynamic code execution, web search, image generation, and voice transcription capabilities.

## 🛠️ Architecture & Modular Structure

The project is structured into clean, maintainable modules:
- **`app.py`**: Gradio web interface, multi-model execution loop, and event handlers.
- **`database.py`**: SQLite database connection, conversation tracking, message logging, and memory storage.
- **`memory.py`**: TF-IDF vectorization, cosine similarity ranking, and automated memory reconciliation.
- **`tools.py`**: `@tool` decorated execution functions (`run_python`, `web_search`, `generate_image`, `speak`) and Whisper transcription.

## ⚙️ Tech Stack

- **Core Framework**: LangChain & LangChain-Groq
- **UI Interface**: Gradio
- **Database**: SQLite
- **Machine Learning & NLP**: scikit-learn (TF-IDF Cosine Similarity), NumPy, Pandas, Matplotlib
- **APIs**: Groq (LLM & Whisper), Pollinations (Image Generation), DuckDuckGo / Wikipedia (Search)

## 🚀 Local Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kamakshi232006/agentic-nexus.git
   cd agentic-nexus

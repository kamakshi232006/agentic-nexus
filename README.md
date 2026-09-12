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

## 🔑 Environment Variables
Create a `.env` file or set the following variables in your hosting environment:
- `GROQ_API_KEY`: Required for LLM execution and Whisper transcription.
- `OPENROUTER_API_KEY`: Optional for fallback or secondary models.

  
[![Live Demo](https://img.shields.io/badge/Status-Live%20on%20Render-brightgreen)](https://agentic-nexus-qtaj.onrender.com)

## 🚀 Local Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kamakshi232006/agentic-nexus.git
   cd agentic-nexus

##Quick Local Run Command: virtual environment setup and dependency installation steps :
```bash
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\Activate on Windows
pip install -r requirements.txt
python app.py

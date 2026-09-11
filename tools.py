#Contains the @tool decorated execution functions, media trackers, and Groq Whisper transcription setup.
import os
import io
import time
import contextlib
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from langchain_core.tools import tool
from groq import Groq

WORK = os.path.abspath("workspace")
os.makedirs(WORK, exist_ok=True)
MEDIA = []        
UPLOADED = [None] 

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
_gc = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

@tool
def run_python(code: str) -> str:
    """Run Python 3 and return its stdout. Use for arithmetic, data analysis, and creating or
    inspecting files. pandas / numpy / matplotlib are imported. If the user attached a file,
    its path is in the variable UPLOADED (e.g. pd.read_csv(UPLOADED)). Save plots with
    plt.savefig('name.png')."""
    before = set(os.listdir(WORK))
    ns = {"pd": __import__("pandas"), "np": np, "plt": plt, "UPLOADED": UPLOADED[0], "os": os}
    buf = io.StringIO()
    cwd = os.getcwd(); os.chdir(WORK)
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, ns)
        err = ""
    except Exception as e:
        err = f"\nError: {type(e).__name__}: {e}"
    finally:
        os.chdir(cwd)
    for i, num in enumerate(plt.get_fignums()):
        f = os.path.join(WORK, f"plot_{int(time.time())}_{i}.png")
        plt.figure(num).savefig(f, bbox_inches="tight"); MEDIA.append(("image", f))
    plt.close("all")
    made = sorted(set(os.listdir(WORK)) - before)
    for fn in made:
        if not fn.startswith("plot_"):
            MEDIA.append(("file", os.path.join(WORK, fn)))
    out = buf.getvalue().strip() or "(no printed output)"
    return out + (f"\n[files created: {', '.join(made)}]" if made else "") + err

@tool
def web_search(query: str) -> str:
    """Search the web for current or factual info (news, prices, people, products, definitions)."""
    try:
        from ddgs import DDGS
        hits = list(DDGS().text(query, max_results=5))
        if hits:
            return "\n".join(f"- {h['title']}: {h['body'][:200]}  <{h['href']}>" for h in hits)
    except Exception as e:
        print("  [web_search] DDG failed ->", str(e)[:80], "- falling back to Wikipedia")
    try:
        ua = {"User-Agent": "GSSS-Bot/1.0"}
        srch = requests.get("https://en.wikipedia.org/w/api.php", headers=ua, timeout=15, params={
            "action": "query", "list": "search", "srsearch": query,
            "format": "json", "srlimit": 3}).json()
        out = []
        for r in srch["query"]["search"]:
            t = r["title"]
            summ = requests.get("https://en.wikipedia.org/api/rest_v1/page/summary/"
                                + t.replace(" ", "_"), headers=ua, timeout=15).json()
            out.append(f"- {t}: {summ.get('extract', '')[:250]}")
        return "(DuckDuckGo unavailable - Wikipedia results)\n" + "\n".join(out) if out else "no results"
    except Exception as e:
        return f"search unavailable: {e}"

@tool
def generate_image(prompt: str) -> str:
    """Create an image from a text description."""
    try:
        url = "https://image.pollinations.ai/prompt/" + requests.utils.quote(prompt) + \
              "?width=768&height=512&nologo=true"
        p = os.path.join(WORK, f"img_{int(time.time())}.png")
        open(p, "wb").write(requests.get(url, timeout=90).content)
        MEDIA.append(("image", p))
        return f"Image created for: {prompt}"
    except Exception as e:
        return f"image error: {e}"

@tool
def speak(text: str) -> str:
    """Convert text to a spoken MP3."""
    try:
        from gtts import gTTS
        p = os.path.join(WORK, f"speak_{int(time.time())}.mp3")
        gTTS(text=text[:800]).save(p)
        MEDIA.append(("audio", p))
        return f"Spoken audio ready ({len(text)} chars)."
    except Exception as e:
        return f"tts error: {e}"

TOOLS = [run_python, web_search, generate_image, speak]

def transcribe(audio_path):
    if not _gc or not audio_path or not os.path.exists(audio_path):
        return ""
    try:
        with open(audio_path, "rb") as f:
            return _gc.audio.transcriptions.create(model="whisper-large-v3", file=f).text.strip()
    except Exception as e:
        print("  [transcribe] failed:", str(e)[:100])
        return ""
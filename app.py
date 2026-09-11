import os
import time
import re
import gradio as gr
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from database import init_db, new_conversation, list_conversations, get_messages, add_message, delete_conversation, rename_conversation, all_mem
from memory import recall, memory_block, reconcile_memory
from tools import TOOLS, MEDIA, UPLOADED, transcribe, speak

init_db()

GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

AGENT_MODELS = [
    ChatGroq(model="qwen/qwen3.8-27b", api_key=GROQ_API_KEY, temperature=0, max_retries=2, request_timeout=45),
    ChatGroq(model="openai/gpt-oss-120b", api_key=GROQ_API_KEY, temperature=0, max_retries=1, request_timeout=45),
]
if OPENROUTER_API_KEY:
    AGENT_MODELS.append(ChatOpenAI(model="minimax/minimax-m2.7:free",
        base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY,
        temperature=0, max_retries=1, request_timeout=45))

def plain_llm(prompt):
    err = None
    for wait in (0, 12, 30):
        if wait:
            time.sleep(wait)
        for m in AGENT_MODELS:
            try:
                return m.invoke(prompt).content
            except Exception as e:
                err = e
    raise err

SYSTEM = (
    "You are a helpful assistant with tools:\n"
    "- run_python: math, data analysis, reading/creating files\n"
    "- web_search: current or factual info that may be newer than your training data\n"
    "- generate_image: any picture, drawing, logo or diagram the user wants\n"
    "- speak: turn text into spoken audio\n"
    "RULES: if the user asks to hear / say / read something aloud, you MUST call speak. "
    "If they want a picture, you MUST call generate_image. For time-sensitive info, call web_search. Be concise.")

def run_agent(agents, messages, max_steps=8):
    err = None
    for wait in (0, 12, 30):
        if wait:
            time.sleep(wait)
        for ag in agents:
            try:
                out = ag.invoke({"messages": messages}, {"recursion_limit": max_steps * 2 + 3})
                trace = [tc["name"] for m in out["messages"] for tc in (getattr(m, "tool_calls", None) or [])]
                return out["messages"][-1].content, trace
            except Exception as e:
                err = e
    raise err

def auto_title(first_msg):
    raw = plain_llm("3-5 word title for a chat starting with this message. Title only, one line.\n\n" + first_msg).strip()
    return raw.splitlines()[0].strip().strip('"').lstrip("#").strip()[:50] or "New chat"

def respond(conversation_id, user_text):
    MEDIA.clear()
    recalled = recall(user_text)
    sys = SYSTEM + ("\n\n" + memory_block(recalled) if memory_block(recalled) else "")
    if UPLOADED[0]:
        sys += f"\n\nThe user attached a file at: {UPLOADED[0]}  (use run_python to read it)."
    history = get_messages(conversation_id)
    agents = [create_agent(m, TOOLS, system_prompt=sys) for m in AGENT_MODELS]
    msgs = [{"role": r, "content": c} for r, c in history] + [{"role": "user", "content": user_text}]

    reply, trace = run_agent(agents, msgs)

    if "speak" not in trace and re.search(r"\b(out ?loud|aloud|read (it|this|that) (aloud|out)|say .*(aloud|out ?loud))\b", user_text, re.I):
        speak.invoke({"text": reply})
        trace.append("speak")

    add_message(conversation_id, "user", user_text)
    add_message(conversation_id, "assistant", reply)
    if not history:
        rename_conversation(conversation_id, auto_title(user_text))
    touched = reconcile_memory(user_text, plain_llm)
    return reply, trace, list(MEDIA), recalled, touched

def mem_rows():
    return [[i, k, c] for i, k, c in all_mem()]

def refresh_list():
    convs = list_conversations()
    choices = [(t, i) for i, t in convs]
    return gr.update(choices=choices, value=(choices[0][1] if choices else None))

def load_conv(cid):
    return [{"role": r, "content": c} for r, c in get_messages(cid)] if cid else []

def on_audio(audio_path, textbox):
    if audio_path:
        return transcribe(audio_path)
    return textbox

def on_file(f):
    UPLOADED[0] = f.name if f else None
    return f"attached: {os.path.basename(UPLOADED[0])}" if UPLOADED[0] else "no file"

def send(cid, user_text, chat):
    if not user_text.strip():
        return chat, "", gr.update(), cid, mem_rows(), "(none)", "(none)", None, None
    if not cid:
        cid = new_conversation()
    reply, trace, media, recalled, touched = respond(cid, user_text)
    chat = load_conv(cid)
    if trace:
        chat.append({"role": "assistant", "content": f"_tools used: {', '.join(trace)}_"
})
    img = next((p for k, p in media if k == "image"), None)
    aud = next((p for k, p in media if k == "audio"), None)
    rec = "\n".join(f"({k}) {c}" for _, k, c in recalled) or "(none)"
    chg = ", ".join(f"#{i}" for i in touched) or "(none)"
    return chat, "", refresh_list(), cid, mem_rows(), rec, chg, img, aud

def new_chat():
    cid = new_conversation()
    return cid, [], refresh_list()

def del_chat(cid):
    if cid:
        delete_conversation(cid)
    convs = list_conversations()
    ncid = convs[0][0] if convs else None
    return ncid, load_conv(ncid), refresh_list()

with gr.Blocks(title="Agentic Nexus - Multi-Tool Agent") as demo:
    gr.Markdown("# Agentic Nexus - Multi-Tool AI Agent with Memory & Voice")
    cur = gr.State(None)
    with gr.Row():
        with gr.Column(scale=1):
            new_btn = gr.Button("New chat", variant="primary")
            chats = gr.Radio(label="Your chats", choices=[])
            del_btn = gr.Button("Delete chat", variant="stop")
            file_in = gr.File(label="Attach a file (csv / xlsx / txt)")
            file_note = gr.Markdown("no file")
        with gr.Column(scale=3):
            box = gr.Chatbot(height=430)
            msg = gr.Textbox(label="Message", placeholder="Type, or use the mic below")
            mic = gr.Audio(sources=["microphone"], type="filepath", label="Speak (Whisper -> text)")
        with gr.Column(scale=2):
            gr.Markdown("### Long-term memory")
            memtab = gr.Dataframe(headers=["id", "kind", "content"], interactive=False, wrap=True)
            recalled_box = gr.Textbox(label="recalled this turn", lines=2)
            changed_box = gr.Textbox(label="memory changed", lines=1)
            out_img = gr.Image(label="image out", height=220)
            out_aud = gr.Audio(label="audio out")

    demo.load(refresh_list, outputs=chats)
    demo.load(mem_rows, outputs=memtab)
    chats.change(lambda c: (c, load_conv(c)), chats, [cur, box])
    new_btn.click(new_chat, outputs=[cur, box, chats])
    del_btn.click(del_chat, cur, [cur, box, chats])
    file_in.change(on_file, file_in, file_note)
    mic.stop_recording(on_audio, [mic, msg], msg)
    msg.submit(send, [cur, msg, box],
               [box, msg, chats, cur, memtab, recalled_box, changed_box, out_img, out_aud])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=False)
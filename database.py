#Handles SQLite connection setup, conversation tracking, message logs, and long-term memory storage.
import sqlite3
import datetime

DB = "mini_chatgpt.db"

def db():
    con = sqlite3.connect(DB)
    con.execute("PRAGMA foreign_keys = ON")
    return con

def init_db():
    with db() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL DEFAULT 'New chat',
                created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, content TEXT NOT NULL,
                created_at TEXT NOT NULL);
        """)

def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")

def new_conversation(title="New chat"):
    with db() as con:
        return con.execute("INSERT INTO conversations(title, created_at) VALUES (?, ?)",
                           (title, _now())).lastrowid

def list_conversations():
    with db() as con:
        return con.execute("SELECT id, title FROM conversations ORDER BY id DESC").fetchall()

def get_messages(cid):
    with db() as con:
        return con.execute("SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id",
                           (cid,)).fetchall()

def add_message(cid, role, content):
    with db() as con:
        con.execute("INSERT INTO messages(conversation_id, role, content, created_at) VALUES (?,?,?,?)",
                    (cid, role, content, _now()))

def rename_conversation(cid, title):
    with db() as con:
        con.execute("UPDATE conversations SET title=? WHERE id=?", (title, cid))

def delete_conversation(cid):
    with db() as con:
        con.execute("DELETE FROM conversations WHERE id=?", (cid,))

def all_mem():
    with db() as con:
        return con.execute("SELECT id, kind, content FROM memories ORDER BY id").fetchall()

def add_mem_db(kind, content):
    with db() as con:
        return con.execute("INSERT INTO memories(kind, content, created_at) VALUES (?,?,?)",
                           (kind, content, _now())).lastrowid

def update_mem_db(i, content):
    with db() as con:
        con.execute("UPDATE memories SET content=?, created_at=? WHERE id=?", (content, _now(), i))

def delete_mem_db(i):
    with db() as con:
        con.execute("DELETE FROM memories WHERE id=?", (i,))
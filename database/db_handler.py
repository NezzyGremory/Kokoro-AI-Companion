# ============================================================
#  KOKORO - Database Handler
#  Menyimpan riwayat percakapan ke SQLite
# ============================================================

import sqlite3
import json
from datetime import datetime
from config import DB_PATH, MAX_HISTORY_MESSAGES


def get_connection() -> sqlite3.Connection:
    """Buka koneksi ke database SQLite."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Buat tabel-tabel yang dibutuhkan jika belum ada."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                role        TEXT    NOT NULL,   -- 'user' atau 'model'
                content     TEXT    NOT NULL,
                timestamp   TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS known_faces (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                name    TEXT    NOT NULL UNIQUE,
                added   TEXT    NOT NULL
            );
        """)


# ─── Conversation History ────────────────────────────────────

def save_message(session_id: str, role: str, content: str) -> None:
    """Simpan satu pesan ke database."""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO conversations (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.now().isoformat()),
        )


def get_history(session_id: str) -> list[dict]:
    """
    Ambil riwayat percakapan terbaru untuk sesi tertentu.
    Mengembalikan list of dict yang kompatibel dengan Gemini API.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM conversations
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, MAX_HISTORY_MESSAGES),
        ).fetchall()

    # Balik urutan agar kronologis (terbaru di akhir)
    history = [{"role": row["role"], "parts": [row["content"]]} for row in reversed(rows)]
    return history


def clear_history(session_id: str) -> None:
    """Hapus seluruh riwayat percakapan untuk sesi tertentu."""
    with get_connection() as conn:
        conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))


# ─── Known Faces ─────────────────────────────────────────────

def add_known_face(name: str) -> None:
    """Daftarkan wajah baru ke database."""
    with get_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO known_faces (name, added) VALUES (?, ?)",
            (name, datetime.now().isoformat()),
        )


def get_known_faces() -> list[str]:
    """Dapatkan daftar nama semua wajah yang sudah dikenal."""
    with get_connection() as conn:
        rows = conn.execute("SELECT name FROM known_faces").fetchall()
    return [row["name"] for row in rows]


# ─── Init on import ──────────────────────────────────────────
init_db()


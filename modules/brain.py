# ============================================================
#  KOKORO - Brain (LLM Interface)
#  Menggunakan Google Gemini API + riwayat percakapan
# ============================================================

import uuid
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL, SYSTEM_PROMPT, ASSISTANT_NAME
from database.db_handler import save_message, get_history, clear_history


class Brain:
    def __init__(self, session_id: str | None = None):
        # Konfigurasi Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self._model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
        )

        # Sesi percakapan
        self.session_id = session_id or str(uuid.uuid4())
        self._chat     = self._restore_chat()

        print(f"[Brain] Session: {self.session_id}")

    # ─── Chat restoration ────────────────────────────────────

    def _restore_chat(self) -> genai.ChatSession:
        """Pulihkan sesi chat dari riwayat database."""
        history = get_history(self.session_id)
        return self._model.start_chat(history=history)

    # ─── Core chat ───────────────────────────────────────────

    def chat(self, user_input: str, context: str = "") -> str:
        """
        Kirim pesan ke Gemini dan kembalikan responsnya.

        Args:
            user_input: Pesan dari pengguna.
            context:    Konteks tambahan (mis. nama wajah yang terdeteksi).

        Returns:
            Teks respons dari asisten.
        """
        if not user_input.strip():
            return ""

        # Sisipkan konteks tambahan jika ada
        message = f"[Konteks: {context}]\n{user_input}" if context else user_input

        try:
            response = self._chat.send_message(message)
            reply = response.text.strip()
        except Exception as e:
            print(f"[Brain] ❌ Error: {e}")
            reply = "Maaf, ada kendala saat memproses permintaanmu. Coba lagi ya!"

        # Simpan ke database
        save_message(self.session_id, "user",  user_input)
        save_message(self.session_id, "model", reply)

        return reply

    def chat_stream(self, user_input: str, context: str = ""):
        """
        Generator: kirim pesan dan streaming respons kata per kata.
        Yield setiap chunk teks.
        """
        if not user_input.strip():
            return

        message = f"[Konteks: {context}]\n{user_input}" if context else user_input

        full_reply = ""
        try:
            response = self._chat.send_message(message, stream=True)
            for chunk in response:
                text = chunk.text
                full_reply += text
                yield text
        except Exception as e:
            print(f"[Brain] ❌ Error: {e}")
            yield "Maaf, ada kendala. Coba lagi ya!"
            return

        # Simpan ke database setelah streaming selesai
        save_message(self.session_id, "user",  user_input)
        save_message(self.session_id, "model", full_reply.strip())

    # ─── Utility ─────────────────────────────────────────────

    def reset(self) -> None:
        """Hapus riwayat percakapan dan mulai sesi baru."""
        clear_history(self.session_id)
        self._chat = self._restore_chat()
        print("[Brain] Riwayat percakapan dihapus.")

    def new_session(self) -> str:
        """Mulai sesi baru dan kembalikan session_id yang baru."""
        self.session_id = str(uuid.uuid4())
        self._chat      = self._restore_chat()
        print(f"[Brain] Sesi baru dimulai: {self.session_id}")
        return self.session_id

    @property
    def history(self) -> list[dict]:
        """Kembalikan riwayat percakapan dari database."""
        return get_history(self.session_id)


# ─── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    brain = Brain()
    while True:
        user = input("Kamu: ").strip()
        if user.lower() in ("exit", "quit", "keluar"):
            break
        reply = brain.chat(user)
        print(f"{ASSISTANT_NAME}: {reply}\n")


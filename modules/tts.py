# ============================================================
#  KOKORO - Text-to-Speech (TTS)
#  Menggunakan pyttsx3 (offline, no internet needed)
# ============================================================

import pyttsx3
import threading
from config import TTS_RATE, TTS_VOLUME


class TTS:
    def __init__(self):
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", TTS_RATE)
        self._engine.setProperty("volume", TTS_VOLUME)
        self._lock = threading.Lock()
        self._is_speaking = False

        # Pilih suara yang tersedia (coba cari suara perempuan)
        self._set_voice()

    def _set_voice(self) -> None:
        """Pilih suara TTS terbaik yang tersedia di sistem."""
        voices = self._engine.getProperty("voices")
        if not voices:
            return

        # Coba cari suara perempuan
        for voice in voices:
            if any(k in voice.name.lower() for k in ("zira", "female", "hazel", "helena")):
                self._engine.setProperty("voice", voice.id)
                return

        # Fallback ke suara pertama
        self._engine.setProperty("voice", voices[0].id)

    # ─── Public API ──────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Ucapkan teks secara sinkron (blocking)."""
        if not text or not text.strip():
            return

        with self._lock:
            self._is_speaking = True
            self._engine.say(text)
            self._engine.runAndWait()
            self._is_speaking = False

    def speak_async(self, text: str) -> threading.Thread:
        """Ucapkan teks di thread terpisah (non-blocking)."""
        thread = threading.Thread(target=self.speak, args=(text,), daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        """Hentikan ucapan yang sedang berjalan."""
        self._engine.stop()
        self._is_speaking = False

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def set_rate(self, rate: int) -> None:
        """Ubah kecepatan bicara (kata per menit)."""
        self._engine.setProperty("rate", rate)

    def set_volume(self, volume: float) -> None:
        """Ubah volume (0.0 – 1.0)."""
        self._engine.setProperty("volume", max(0.0, min(1.0, volume)))

    def list_voices(self) -> list[str]:
        """Tampilkan daftar suara yang tersedia."""
        return [v.name for v in self._engine.getProperty("voices")]


# ─── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    tts = TTS()
    print("Voices available:", tts.list_voices())
    tts.speak("Halo! Saya Kokoro, asisten AI kamu. Siap membantu!")


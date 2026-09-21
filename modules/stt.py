# ============================================================
#  KOKORO - Speech-to-Text (STT)
#  Menggunakan Faster-Whisper (lokal, akurat, cepat)
# ============================================================

import numpy as np
import sounddevice as sd
import threading
from faster_whisper import WhisperModel
from config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_LANGUAGE


# Sample rate untuk audio recording
SAMPLE_RATE   = 16_000  # Hz — dibutuhkan Whisper
CHUNK_DURATION = 0.5    # detik per chunk saat streaming


class STT:
    def __init__(self):
        print(f"[STT] Loading Whisper model '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE}...")
        self._model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type="int8",   # hemat RAM, tetap akurat
        )
        self._stop_event = threading.Event()
        print("[STT] Model loaded ✓")

    # ─── Core transcription ──────────────────────────────────

    def transcribe_file(self, audio_path: str) -> str:
        """Transkripsi file audio (.wav / .mp3 / dll)."""
        segments, _ = self._model.transcribe(
            audio_path,
            language=WHISPER_LANGUAGE or None,
            beam_size=5,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def transcribe_array(self, audio: np.ndarray) -> str:
        """Transkripsi dari numpy array float32 (16 kHz, mono)."""
        segments, _ = self._model.transcribe(
            audio,
            language=WHISPER_LANGUAGE or None,
            beam_size=5,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    # ─── Live recording ──────────────────────────────────────

    def listen(self, duration: float = 5.0, silence_threshold: float = 0.01) -> str:
        """
        Rekam audio dari mikrofon selama `duration` detik,
        lalu transkripsi dan kembalikan teksnya.
        """
        print(f"[STT] 🎤 Mendengarkan selama {duration} detik...")
        audio = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()

        audio_flat = audio.flatten()

        # Cek apakah ada suara (hindari transkripsi keheningan)
        if np.abs(audio_flat).mean() < silence_threshold:
            print("[STT] Tidak terdeteksi suara.")
            return ""

        text = self.transcribe_array(audio_flat)
        print(f"[STT] Transkripsi: {text!r}")
        return text

    def listen_continuous(self, callback, stop_event: threading.Event | None = None) -> None:
        """
        Rekam dan transkripsi secara terus-menerus dalam loop.
        `callback(text: str)` dipanggil setiap kali ada hasil transkripsi.
        Loop berhenti saat `stop_event` di-set.
        """
        stop = stop_event or self._stop_event
        print("[STT] Mulai mendengarkan terus-menerus. Tekan Ctrl+C untuk berhenti.")

        buffer_size = int(3.0 * SAMPLE_RATE)  # 3 detik per chunk

        def audio_callback(indata, frames, time_info, status):
            if stop.is_set():
                raise sd.CallbackStop()

            audio_flat = indata.flatten()
            if np.abs(audio_flat).mean() < 0.01:
                return

            text = self.transcribe_array(audio_flat)
            if text:
                callback(text)

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=buffer_size,
                callback=audio_callback,
            ):
                stop.wait()
        except KeyboardInterrupt:
            pass

    def stop_listening(self) -> None:
        """Hentikan `listen_continuous`."""
        self._stop_event.set()


# ─── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    stt = STT()
    result = stt.listen(duration=5)
    print("Hasil:", result)


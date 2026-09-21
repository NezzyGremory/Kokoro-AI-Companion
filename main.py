# ============================================================
#  KOKORO - Main Entry Point
#  AI Assistant dengan suara, penglihatan, dan memori
# ============================================================

import threading
import sys

from config import ASSISTANT_NAME
from modules.brain  import Brain
from modules.tts    import TTS
from modules.stt    import STT
from modules.vision import Vision


def greet(brain: Brain, tts: TTS, vision: Vision) -> None:
    """Sapa pengguna saat pertama kali menyala."""
    names = vision.snapshot_and_identify()

    if names:
        known = [n for n in names if n != "Unknown"]
        if known:
            context = f"Wajah yang terdeteksi: {', '.join(known)}"
            greeting = brain.chat(f"Sapa {', '.join(known)} dengan hangat!", context=context)
        else:
            greeting = brain.chat("Sapa pengguna baru yang wajahnya belum dikenal.")
    else:
        greeting = brain.chat("Ucapkan salam pembuka singkat.")

    print(f"\n{ASSISTANT_NAME}: {greeting}\n")
    tts.speak(greeting)


def run_voice_mode(brain: Brain, tts: TTS, stt: STT) -> None:
    """Mode utama: dengarkan → proses → balas dengan suara."""
    print("=" * 50)
    print(f"  {ASSISTANT_NAME} — Voice Mode")
    print("  Ucapkan 'keluar' atau 'exit' untuk berhenti.")
    print("=" * 50)

    while True:
        # Dengarkan pengguna
        user_input = stt.listen(duration=5)
        if not user_input:
            continue

        print(f"\nKamu : {user_input}")

        # Cek perintah keluar
        if any(w in user_input.lower() for w in ("keluar", "exit", "quit", "bye")):
            farewell = brain.chat("Ucapkan salam perpisahan yang hangat.")
            print(f"{ASSISTANT_NAME}: {farewell}")
            tts.speak(farewell)
            break

        # Perintah reset memori
        if any(w in user_input.lower() for w in ("reset", "hapus memori", "lupa semua")):
            brain.reset()
            msg = "Oke, saya sudah lupa semua percakapan kita. Mulai fresh!"
            print(f"{ASSISTANT_NAME}: {msg}")
            tts.speak(msg)
            continue

        # Kirim ke brain
        reply = brain.chat(user_input)
        print(f"{ASSISTANT_NAME}: {reply}\n")
        tts.speak(reply)


def run_text_mode(brain: Brain, tts: TTS) -> None:
    """Mode teks: input manual dari terminal."""
    print("=" * 50)
    print(f"  {ASSISTANT_NAME} — Text Mode")
    print("  Ketik 'keluar' untuk berhenti.")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nKamu: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() in ("keluar", "exit", "quit"):
            farewell = brain.chat("Ucapkan salam perpisahan yang hangat.")
            print(f"{ASSISTANT_NAME}: {farewell}")
            tts.speak(farewell)
            break

        if user_input.lower() in ("reset", "hapus memori"):
            brain.reset()
            print(f"{ASSISTANT_NAME}: Memori dihapus. Mulai dari awal!")
            continue

        reply = brain.chat(user_input)
        print(f"{ASSISTANT_NAME}: {reply}")
        tts.speak_async(reply)


def main() -> None:
    print(f"\n🌸 Memuat {ASSISTANT_NAME}...\n")

    # Inisialisasi semua modul
    brain  = Brain()
    tts    = TTS()
    vision = Vision()

    # Pilih mode
    mode = "text"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()  # python main.py voice / text

    if mode == "voice":
        stt = STT()
        greet(brain, tts, vision)
        run_voice_mode(brain, tts, stt)
    else:
        greet(brain, tts, vision)
        run_text_mode(brain, tts)

    print("\n👋 Sampai jumpa!")


if __name__ == "__main__":
    main()


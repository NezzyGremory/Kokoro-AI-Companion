# ============================================================
#  KOKORO - Vision Module
#  Face detection & recognition menggunakan OpenCV + face_recognition
# ============================================================

import os
import cv2
import numpy as np
import face_recognition
from config import FACE_DATASET_DIR, CAMERA_INDEX, FACE_TOLERANCE
from database.db_handler import add_known_face, get_known_faces


class Vision:
    def __init__(self):
        self._known_encodings: list[np.ndarray] = []
        self._known_names: list[str] = []
        self._cap: cv2.VideoCapture | None = None
        self.load_known_faces()

    # ─── Face database ───────────────────────────────────────

    def load_known_faces(self) -> None:
        """
        Muat semua wajah dari folder dataset.
        Struktur: dataset/faces/<nama>/<foto.jpg>
        """
        self._known_encodings.clear()
        self._known_names.clear()

        if not os.path.exists(FACE_DATASET_DIR):
            os.makedirs(FACE_DATASET_DIR, exist_ok=True)
            return

        for person_name in os.listdir(FACE_DATASET_DIR):
            person_dir = os.path.join(FACE_DATASET_DIR, person_name)
            if not os.path.isdir(person_dir):
                continue

            for filename in os.listdir(person_dir):
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                img_path = os.path.join(person_dir, filename)
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    self._known_encodings.append(encodings[0])
                    self._known_names.append(person_name)

        print(f"[Vision] Loaded {len(self._known_names)} face(s): {set(self._known_names)}")

    def register_face_from_camera(self, name: str, num_samples: int = 5) -> bool:
        """
        Tangkap beberapa foto wajah dari kamera dan simpan ke dataset.
        Returns True jika berhasil.
        """
        save_dir = os.path.join(FACE_DATASET_DIR, name)
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print("[Vision] ❌ Kamera tidak bisa dibuka.")
            return False

        captured = 0
        print(f"[Vision] 📸 Menangkap {num_samples} foto untuk '{name}'...")

        while captured < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if face_locations:
                img_path = os.path.join(save_dir, f"{captured}.jpg")
                cv2.imwrite(img_path, frame)
                captured += 1
                print(f"[Vision]  Sampel {captured}/{num_samples} disimpan.")
                cv2.waitKey(500)  # jeda antar foto

            cv2.imshow("Registrasi Wajah — tekan Q untuk batal", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        if captured == num_samples:
            add_known_face(name)
            self.load_known_faces()
            return True
        return False

    # ─── Face recognition ────────────────────────────────────

    def identify_face_from_frame(self, frame: np.ndarray) -> list[tuple[str, tuple]]:
        """
        Identifikasi wajah di frame kamera.
        Returns list of (nama, bounding_box_tuple).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations  = face_recognition.face_locations(rgb)
        encodings  = face_recognition.face_encodings(rgb, locations)

        results = []
        for encoding, location in zip(encodings, locations):
            name = "Unknown"
            if self._known_encodings:
                matches   = face_recognition.compare_faces(
                    self._known_encodings, encoding, tolerance=FACE_TOLERANCE
                )
                distances = face_recognition.face_distance(self._known_encodings, encoding)

                if matches:
                    best_idx = int(np.argmin(distances))
                    if matches[best_idx]:
                        name = self._known_names[best_idx]

            results.append((name, location))
        return results

    def snapshot_and_identify(self) -> list[str]:
        """
        Ambil satu frame dari kamera, identifikasi wajah, kembalikan daftar nama.
        """
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            return []

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return []

        faces = self.identify_face_from_frame(frame)
        names = [name for name, _ in faces]
        return names

    # ─── Live preview ─────────────────────────────────────────

    def show_live_recognition(self) -> None:
        """
        Tampilkan preview kamera secara live dengan bounding box dan nama.
        Tekan Q untuk keluar.
        """
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print("[Vision] ❌ Kamera tidak bisa dibuka.")
            return

        print("[Vision] Live recognition aktif. Tekan Q untuk berhenti.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = self.identify_face_from_frame(frame)

            for name, (top, right, bottom, left) in faces:
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(
                    frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2,
                )

            cv2.imshow("KOKORO — Live Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


# ─── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    vision = Vision()
    vision.show_live_recognition()


"""
From-scratch eigenfaces face recognizer (PCA on raw pixel vectors).
"""
import os
import numpy as np  # type: ignore[import-not-found]
from utils import preprocess_image, IMG_SIZE

N_PIXELS = IMG_SIZE[0] * IMG_SIZE[1]
DATASET_DIR = "dataset"

# Tune this after re-running preprocess.py with the fixed (matching) resize
# logic -- distances will shift once training/inference preprocessing agree.
THRESHOLD = 3000.0


class FaceRecognizer:
    def __init__(self, dataset_dir=DATASET_DIR, threshold=THRESHOLD):
        self.threshold = threshold

        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(
                f"'{dataset_dir}' not found. Add photos to 'raw_dataset' and "
                f"run preprocess.py first."
            )

        image_files = sorted(
            f for f in os.listdir(dataset_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        )
        if not image_files:
            raise ValueError(f"No images found in '{dataset_dir}'. Run preprocess.py first.")

        self.labels = []
        data_matrix = np.zeros((N_PIXELS, len(image_files)))
        for i, filename in enumerate(image_files):
            data_matrix[:, i] = self._load_vector(os.path.join(dataset_dir, filename))
            # Assumes filenames like "alice_01.jpg" -> label "alice"
            self.labels.append(filename.split('_')[0])

        self.num_images = len(image_files)
        self.mean_face = np.mean(data_matrix, axis=1, keepdims=True)
        A = data_matrix - self.mean_face

        # A.T @ A is symmetric -> use eigh, not eig.
        # eig() doesn't assume symmetry and can return complex eigenvalues/
        # eigenvectors due to floating point noise, which silently corrupts
        # the eigenfaces below (norms, sorting, etc. all behave oddly on
        # complex numbers that should have been real).
        L = A.T @ A
        eigenvalues, eigenvectors_L = np.linalg.eigh(L)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors_L = eigenvectors_L[:, sorted_indices]

        # Mean-subtracted data has rank <= num_images - 1, so the smallest
        # eigenvalue(s) are ~0 numerical noise, not real variance directions.
        # Normalizing their eigenvectors would inject noise into the basis.
        keep = eigenvalues > 1e-6
        eigenvectors_L = eigenvectors_L[:, keep]

        eigenfaces = A @ eigenvectors_L
        eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)

        self.eigenfaces = eigenfaces
        self.dataset_weights = eigenfaces.T @ A

    @staticmethod
    def _load_vector(path):
        img = preprocess_image(path)
        return np.array(img, dtype=np.float64).flatten()

    def identify(self, test_path, verbose=False):
        test_vector = self._load_vector(test_path).reshape(-1, 1)
        normalized_test = test_vector - self.mean_face
        test_weights = self.eigenfaces.T @ normalized_test

        distances = np.linalg.norm(self.dataset_weights - test_weights, axis=0)

        if verbose:
            for label, dist in zip(self.labels, distances):
                print(f"Distance to {label}: {dist:.2f}")

        match_index = int(np.argmin(distances))
        min_distance = float(distances[match_index])

        if min_distance > self.threshold:
            return "Access Denied", min_distance
        return self.labels[match_index], min_distance


# --- Module-level convenience API -------------------------------------
# Kept so existing code (`from app import identify`) doesn't need to change.
# The model is now built lazily on first use instead of at import time, so
# importing this module no longer crashes just because the dataset is
# missing -- you only get an error when you actually try to identify().
_recognizer = None


def _get_recognizer():
    global _recognizer
    if _recognizer is None:
        _recognizer = FaceRecognizer()
    return _recognizer


def identify(test_path):
    name, _ = _get_recognizer().identify(test_path)
    return name


def load_vector(path):
    return _get_recognizer()._load_vector(path)


if __name__ == "__main__":
    # Quick sanity check: run every training image back through identify()
    # and confirm it matches itself. Good smoke test after any threshold
    # or preprocessing change.
    rec = FaceRecognizer()
    correct = 0
    for filename in sorted(os.listdir(DATASET_DIR)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        path = os.path.join(DATASET_DIR, filename)
        expected = filename.split('_')[0]
        name, dist = rec.identify(path, verbose=False)
        ok = (name == expected)
        correct += ok
        print(f"{filename}: predicted={name} expected={expected} dist={dist:.1f} {'OK' if ok else 'MISMATCH'}")
    print(f"\n{correct}/{rec.num_images} self-matched correctly.")
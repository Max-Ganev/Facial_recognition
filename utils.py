"""
Shared image preprocessing.

Every entry point (preprocess.py, app.py, gui.py, video_view.py) must use
THIS function to prepare images. If training and inference use different
resize/resampling logic, the resulting pixel vectors won't be comparable
and matching accuracy will silently degrade.
"""
from PIL import Image

IMG_SIZE = (100, 100)


def preprocess_image(source, size=IMG_SIZE):
    """
    Convert an image to grayscale and resize it consistently.

    `source` can be a file path/str, or an already-open PIL Image
    (useful for webcam frames, which come from a numpy array, not a file).
    Returns a PIL Image.
    """
    if isinstance(source, Image.Image):
        img = source
    else:
        img = Image.open(source)

    img_gray = img.convert('L')
    img_resized = img_gray.resize(size, Image.Resampling.LANCZOS)
    return img_resized
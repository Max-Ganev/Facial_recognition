import os
import re
import shutil
import sys
from collections import defaultdict
from utils import preprocess_image


def extract_label(filename):
    """
    Turn a raw filename into a clean person-label, regardless of naming
    convention. Handles things like:
        "Max (1).png"   -> "Max"
        "Max_1.png"      -> "Max"
        "Max-1.png"      -> "Max"
        "Max.png"        -> "Max"
        "alice_smith_02.jpg" -> "alice_smith"   (trailing number stripped)

    This matters because the recognizer groups training photos by label --
    if every "Max (n).png" file were treated as a different filename with no
    normalization, they'd each be read as a *different* person instead of
    multiple photos of the same one.
    """
    name = os.path.splitext(filename)[0]

    # Strip a trailing " (n)" -- Windows adds this automatically when you
    # copy/save files with the same name multiple times.
    name = re.sub(r'\s*\(\d+\)\s*$', '', name)

    # Strip a trailing separator + number, e.g. "Max_1", "Max-1", "Max 1"
    name = re.sub(r'[\s_-]+\d+\s*$', '', name)

    name = name.strip()
    return name if name else "unknown"


def clean_and_resize(source_dir, output_dir):
    # Wipe and recreate output_dir every run. Without this, old files from a
    # previous naming scheme (or deleted raw_dataset photos) stick around
    # and get loaded alongside the new ones -- silently doubling/corrupting
    # the dataset the recognizer trains on.
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    processed = 0
    skipped = 0
    counts = defaultdict(int)  # per-label counter so output filenames stay unique

    source_files = sorted(
        f for f in os.listdir(source_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    )

    for filename in source_files:
        src_path = os.path.join(source_dir, filename)
        try:
            label = extract_label(filename)
            ext = os.path.splitext(filename)[1].lower()

            counts[label] += 1
            dest_filename = f"{label}_{counts[label]}{ext}"
            dest_path = os.path.join(output_dir, dest_filename)

            resized = preprocess_image(src_path)
            resized.save(dest_path)

            print(f"Success: {filename} -> {dest_filename}  (label: {label})")
            processed += 1
        except Exception as e:
            # One corrupt file no longer kills the whole batch.
            print(f"[!] Skipped {filename}: {e}")
            skipped += 1

    print(f"\nDone. Processed {processed} image(s), skipped {skipped}.")
    print(f"Labels found: {dict(counts)}")
    return processed, skipped


if __name__ == "__main__":
    if os.path.exists("raw_dataset"):
        print("Processing training folder...")
        count, _ = clean_and_resize("raw_dataset", "dataset")
        if count == 0:
            print("[!] No valid images were processed. Check 'raw_dataset' contents.")
            sys.exit(1)
    else:
        print("[!] 'raw_dataset' directory not found. Please create it and add training images.")
        sys.exit(1)
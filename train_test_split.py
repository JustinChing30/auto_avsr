import glob
import os
import re
# Import TextTransform from Auto-AVSR transforms
from datamodule.transforms import TextTransform

project_root = r"E:\Media-Files\Real-Stuff\School\Code-Files\Python Files\Auto-AVSR"
preprocess_root = os.path.join(project_root, "ouluvs2-preprocess")
text_dir = os.path.join(preprocess_root, "custom", "custom_text_seg16s")
labels_dir = os.path.join(preprocess_root, "labels")
os.makedirs(labels_dir, exist_ok=True)

# Initialize SentencePiece tokenizer transformer
text_transform = TextTransform()

train_speakers = {f"s{i}" for i in range(1, 41)}
val_speakers = {f"s{i}" for i in range(41, 48)}
test_speakers = {f"s{i}" for i in range(48, 54)}


def get_split(filename):
    match = re.search(r"(?:^|_)(s\d+)(?:_|\.|$)", filename)
    if not match:
        return None
    spk = match.group(1)
    if spk in train_speakers:
        return "train"
    if spk in val_speakers:
        return "val"
    if spk in test_speakers:
        return "test"
    return None


splits = {"train": [], "val": [], "test": []}

for txt_path in glob.glob(os.path.join(text_dir, "*.txt")):
    fname = os.path.basename(txt_path)
    base_name = os.path.splitext(fname)[0]
    split = get_split(fname)

    if split:
        with open(txt_path, "r", encoding="utf-8") as f:
            transcript = f.read().strip()

        # Tokenize the text into space-separated integer IDs
        token_tensor = text_transform.tokenize(transcript)
        token_ids_str = " ".join(str(tid.item()) for tid in token_tensor)

        rel_video_path = f"custom/custom_video_seg16s/{base_name}.mp4"
        splits[split].append(f"ouluvs2,{rel_video_path},100,{token_ids_str}\n")

for split_name, lines in splits.items():
    csv_file = os.path.join(
        labels_dir, f"ouluvs2_{split_name}_transcript_lengths.csv"
    )
    with open(csv_file, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Wrote {len(lines)} items to {csv_file}")
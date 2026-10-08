import glob
import os
import re

# Dataset Paths
project_root = (
    r"E:\Media-Files\Real-Stuff\School\Code-Files\Python Files\Auto-AVSR"
)
digit_phrase_file = os.path.join(
    project_root, r"OuluVS2\OuluVS2-zip\transcript_digit_phrase"
)
sentence_dir = os.path.join(
    project_root, r"OuluVS2\OuluVS2-zip\transcript_sentence"
)

preprocess_root = os.path.join(project_root, "ouluvs2-preprocess")
text_dir = os.path.join(preprocess_root, "custom", "custom_text_seg16s")
labels_dir = os.path.join(preprocess_root, "labels")


def clean_text(text):
    """Remove trailing counts like (x3), strip punctuation, and uppercase."""
    text = re.sub(r"\(\s*x\s*\d+\s*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.strip().upper().split())


digit_sequences = []
phrases = []

# 1. Load Digits and Phrases
if os.path.exists(digit_phrase_file):
    with open(digit_phrase_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = [clean_text(l) for l in f.read().splitlines() if l.strip()]

    valid_lines = [l for l in lines if l]

    # First 10 lines = 10 Digit Sequences (1 line per sequence)
    digit_sequences = valid_lines[:10]

    # Next 10 lines (indices 10..19) = 10 Short Phrases
    phrases = valid_lines[10:20]

print("--- VERIFICATION OF LOADED DIGITS ---")
for idx, seq in enumerate(digit_sequences):
    print(f"Digit Seq {idx+1}: {seq}")

print("\n--- VERIFICATION OF LOADED PHRASES ---")
for idx, ph in enumerate(phrases):
    print(f"Phrase {idx+1}: {ph}")

# 2. Load Speaker-Dependent Sentences
speaker_sentences = {}

for subject_id in range(1, 54):
    spk_key = f"s{subject_id}"
    spk_file = os.path.join(sentence_dir, spk_key)

    if os.path.exists(spk_file):
        with open(spk_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = [
                clean_text(l) for l in f.read().splitlines() if l.strip()
            ]
        speaker_sentences[spk_key] = lines[:10]


def get_transcript(filename):
    sub_match = re.search(r"(?:^|_)(s\d+)(?:_|\.|$)", filename)
    utt_match = re.search(r"(?:^|_)(u\d+)(?:_|\.|$)", filename)

    sub = sub_match.group(1) if sub_match else None
    utt_str = utt_match.group(1) if utt_match else None

    if not utt_str:
        return ""

    utt_num = int(utt_str.replace("u", ""))

    # u1 to u30: Digits
    if 1 <= utt_num <= 30:
        seq_idx = (utt_num - 1) // 3
        if seq_idx < len(digit_sequences):
            return digit_sequences[seq_idx]

    # u31 to u60: Phrases
    elif 31 <= utt_num <= 60:
        phrase_idx = (utt_num - 31) // 3
        if phrase_idx < len(phrases):
            return phrases[phrase_idx]

    # u61 to u70: Sentences
    elif 61 <= utt_num <= 70:
        sent_idx = utt_num - 61
        if sub and sub in speaker_sentences:
            spk_sents = speaker_sentences[sub]
            if sent_idx < len(spk_sents):
                return spk_sents[sent_idx]

    return ""


# 3. Update individual text label files
txt_files = glob.glob(os.path.join(text_dir, "*.txt"))
updated = 0

for filepath in txt_files:
    fname = os.path.basename(filepath)
    transcript = get_transcript(fname)

    if transcript:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(transcript)
        updated += 1

print(f"\nUpdated {updated}/{len(txt_files)} text files in custom_text_seg16s.")

# 4. Update the master CSV manifest file
csv_path = os.path.join(
    labels_dir, "custom_train_transcript_lengths_seg16s.csv"
)

if os.path.exists(csv_path):
    updated_lines = []
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split(",")
        if len(parts) >= 2:
            rel_path, frame_len = parts[0], parts[1]
            fname = os.path.basename(rel_path)
            transcript = get_transcript(fname)
            updated_lines.append(f"{rel_path},{frame_len},{transcript}\n")
        else:
            updated_lines.append(line)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.writelines(updated_lines)

    print(f"Updated master CSV manifest at: {csv_path}")
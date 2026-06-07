"""
(Shared) preprocessing utilities for NusaTranslation Sentiment Analysis.
Dipakai di semua notebook: classical baseline, fine-tuning, dan evaluasi.
"""

import re
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict


# Constants

LANGUAGES   = ['jav', 'min', 'sun']
LABEL_MAP   = {0: 'negative', 2: 'positive'}   # label 1 (neutral) tidak ada
NUM_LABELS  = 2

# Mapping label asli (0, 2) ke index model (0, 1)
LABEL_REMAP = {0: 0, 2: 1}
LABEL_REMAP_INVERSE = {0: 'negative', 1: 'positive'}


#Text Cleaning

def clean_text(text: str) -> str:
    """
    Membersihkan teks dari artifcats sosial media.
    Urutan operasi penting: URL dulu sebelum karakter khusus.
    """
    # Hapus URL
    text = re.sub(r'http\S+|www\S+', '', text)

    # Hapus mention (@username)
    text = re.sub(r'@\w+', '', text)

    # Hapus hashtag simbol tapi pertahankan kata-katanya
    text = re.sub(r'#(\w+)', r'\1', text)

    # Hapus karakter non-alfanumerik kecuali spasi dan tanda baca dasar
    text = re.sub(r'[^\w\s]', ' ', text)

    # Normalisasi spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def remap_labels(label: int) -> int:
    """
    Remap label asli dataset (0, 2) ke index model (0, 1).
    Diperlukan karena HuggingFace classification head
    mengharapkan label kontinu mulai dari 0.
    """
    return LABEL_REMAP[label]


# ── Dataset Processing ─────────────────────────────────────────────────────────

def preprocess_dataset(dataset_dict: DatasetDict) -> DatasetDict:
    """
    Terapkan clean_text dan remap_labels ke seluruh split
    (train, validation, test) dari satu bahasa.
    """
    def process_batch(batch):
        batch['text']  = [clean_text(t) for t in batch['text']]
        batch['label'] = [remap_labels(l) for l in batch['label']]
        return batch

    return dataset_dict.map(process_batch, batched=True)


def get_splits_as_df(dataset_dict: DatasetDict) -> dict[str, pd.DataFrame]:
    """
    Konversi DatasetDict ke dict of DataFrames.
    Useful untuk classical ML pipeline.
    """
    return {
        split: pd.DataFrame(dataset_dict[split])
        for split in ['train', 'validation', 'test']
    }


# Class Weight 

def compute_class_weights(labels: list | np.ndarray) -> dict:
    """
    Hitung class weight untuk handle imbalance ~70:30.
    Formula: total_samples / (n_classes * count_per_class)

    Return dict {class_index: weight} untuk sklearn,
    dan tensor weights untuk PyTorch loss function.
    """
    labels    = np.array(labels)
    classes   = np.unique(labels)
    n_samples = len(labels)
    n_classes = len(classes)

    weights = {}
    for c in classes:
        count      = np.sum(labels == c)
        weights[c] = n_samples / (n_classes * count)

    return weights


# Quick Test

if __name__ == '__main__':
    test_texts = [
        "Bar dijawab dibusak http://ask.fm/a/bo26h59f @user123 #happy",
        "  Abis dijawek   diapuih,kawan macam apo ?  ",
        "Beres dijawab di hapus, babaturan siga naon ?",
    ]

    print("=== Text Cleaning Test ===")
    for t in test_texts:
        print(f"  Before : {t}")
        print(f"  After  : {clean_text(t)}")
        print()

    print("=== Label Remap Test ===")
    for original, expected in [(0, 0), (2, 1)]:
        result = remap_labels(original)
        status = "OK" if result == expected else "FAIL"
        print(f"  [{status}] label {original} -> {result}")

    print("\n=== Class Weight Test ===")
    dummy_labels = [0]*2382 + [1]*1018
    weights = compute_class_weights(dummy_labels)
    print(f"  Class weights: {weights}")
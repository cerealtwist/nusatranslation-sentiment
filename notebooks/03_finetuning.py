import sys
sys.path.append('../')

import os
import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score

from src.preprocessing import (
    LANGUAGES,
    NUM_LABELS,
    preprocess_dataset,
)

# Konfirmasi device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

import sys
sys.path.append('../')

import os
import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score

from src.preprocessing import (
    LANGUAGES,
    NUM_LABELS,
    preprocess_dataset,
)

# Konfirmasi device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

MODELS = {
    'IndoBERT': 'indobenchmark/indobert-base-p2',
    'mBERT'   : 'google-bert/bert-base-multilingual-cased',
    'XLM-R'   : 'FacebookAI/xlm-roberta-base'
}

datasets_per_lang = {}

for lang in LANGUAGES:
    raw = load_dataset(
        'indonlp/nusatranslation_senti',
        name=f'nusatranslation_senti_{lang}_nusantara_text',
        trust_remote_code=True
    )
    datasets_per_lang[lang] = preprocess_dataset(raw)
    print(f"[{lang}] loaded and preprocessed.")

def tokenize_dataset(dataset_dict, tokenizer, max_length=128):
    def tokenize_batch(batch):
        return tokenizer(
            batch['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
    return dataset_dict.map(tokenize_batch, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average='macro')
    return {'macro_f1': macro_f1}

results = []
os.makedirs('../results/checkpoints', exist_ok=True)

for model_name, model_path in MODELS.items():
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    for lang in LANGUAGES:
        print(f"\n--- [{lang}] ---")

        # Tokenisasi
        tokenized = tokenize_dataset(
            datasets_per_lang[lang], 
            tokenizer
        )

        # Inisialisasi model baru tiap kombinasi
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=True
        )
        
        from peft import get_peft_model, LoraConfig, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        output_dir = f'../results/checkpoints/{model_name}_{lang}'

        training_args = TrainingArguments(
            output_dir=output_dir,

            # Hyperparameter dari paper NusaWrites
            learning_rate=2e-4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,  # efektif batch 32
            num_train_epochs=5,

            # Evaluasi dan early stopping
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='macro_f1',
            greater_is_better=True,

            # Logging
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            report_to='none',

            # Reproducibility
            seed=SEED,

            # CPU optimization
            use_cpu=False,
            fp16=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized['train'],
            eval_dataset=tokenized['validation'],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        trainer.train()

        # Evaluasi di test set
        test_output = trainer.predict(tokenized['test'])
        preds       = np.argmax(test_output.predictions, axis=-1)
        labels      = test_output.label_ids
        macro_f1    = f1_score(labels, preds, average='macro')

        print(f"[{model_name}][{lang}] Test Macro F1: {macro_f1:.4f}")

        results.append({
            'model'   : model_name,
            'language': lang,
            'macro_f1': round(macro_f1, 4)
        })

        # Simpan hasil sementara tiap run selesai
        # supaya kalau interrupted, hasil sebelumnya tidak hilang
        try:
            pd.DataFrame(results).to_csv(
                '../results/dl_results.csv', index=False
            )
        except PermissionError:
            print("\n[PERINGATAN] File dl_results.csv sedang dibuka (mungkin di Excel). Tidak bisa menyimpan hasil sementara. Mohon tutup file tersebut!\n")

        # Hapus model dari memory sebelum load model berikutnya
        del model
        torch.cuda.empty_cache() if DEVICE == 'cuda' else None

print("\nSemua training selesai.")
print(pd.DataFrame(results).to_string(index=False))

dl_results = pd.read_csv('../results/dl_results.csv')

pivot = dl_results.pivot(
    index='language',
    columns='model',
    values='macro_f1'
)

print("=== Deep Learning Results -- Test Set Macro F1 ===")
print(pivot.to_string())
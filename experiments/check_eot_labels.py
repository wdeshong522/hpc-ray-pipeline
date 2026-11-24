"""Does the trainer learn to emit the end-of-turn token?

Runs the same label-masking collator the trainer uses on four real examples,
once with pad token = eos (the old code's choice) and once with our pad token.
Counts end-of-turn tokens in the input and how many keep a training label.
CPU only; loads just the tokenizer.

Usage (repo root):  python experiments/check_eot_labels.py MODEL DATA_DIR
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer, DataCollatorForLanguageModeling  # noqa: E402

from src.train.data import (  # noqa: E402
    load_records, ordered_pool, take, to_messages)
from src.train.model import load_tokenizer  # noqa: E402

EOT = "<|eot_id|>"


def report(label, tokenizer, texts):
    batch = [{"input_ids": tokenizer(t, add_special_tokens=False)["input_ids"]}
             for t in texts]
    out = DataCollatorForLanguageModeling(tokenizer, mlm=False)(batch)
    eot_id = tokenizer.convert_tokens_to_ids(EOT)
    is_eot = out["input_ids"] == eot_id
    kept = (is_eot & (out["labels"] != -100)).sum().item()
    print(f"{label:10s} eos={tokenizer.eos_token} pad={tokenizer.pad_token} "
          f"eot_in_input={is_eot.sum().item()} eot_with_label={kept}")


model_name, data_dir = sys.argv[1], sys.argv[2]
records, _ = load_records(data_dir)
sample = take(ordered_pool(records), 4)

base = AutoTokenizer.from_pretrained(model_name)
texts = [base.apply_chat_template(to_messages(r), tokenize=False,
                                  add_generation_prompt=False)
         for r in sample]

old = AutoTokenizer.from_pretrained(model_name)
old.pad_token = old.eos_token
report("pad=eos", old, texts)
report("pad=ours", load_tokenizer(model_name), texts)

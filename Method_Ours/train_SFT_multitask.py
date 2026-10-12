"""
Joint hierarchical policy optimization (single shared phi'):

  L_joint = L_act(phi') + lambda * L_qry(phi')

  L_act = KL( p_hat(a|x) || p_hat_phi'(a|x) ),
    p_hat(a|x) proportional to exp(s(a)/beta) over actions in A.
  L_qry = -sum_i log p_phi'(q_i | q_<i, a, x)  (teacher query supervision under the gen branch).

Dataset JSON (dict keyed by sample id), fields per item:
  - question: str
  - history: str (private context; fills {profile} in QUESTION_USER_PROMPT)
  - score: { "avg": [s_PLAN, s_FACT, s_REFLECT, s_ORIGIN] } (four floats)
  - selected_augmentation: clarify | knowledge | verify | origin
  - sft_sup: str (teacher target for the refinement branch; omit or empty if origin)

Requires: transformers, peft, torch.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.formetters import (
    QUESTION_USER_PROMPT,
    SCORE_SYSTEM_PROMPT,
    _CLARIFY_SYSTEM_PROMPT,
    _KNOWLEDGE_SYSTEM_PROMPT,
    _VERIFY_SYSTEM_PROMPT,
)
from utils.custom_llm import build_chat_prompt

LABEL_WORDS = ["PLAN", "FACT", "REFLECT", "ORIGIN"]
LABEL_TOKEN_IDS: List[int] = []

AUG_TO_SYSTEM_PROMPT = {
    "clarify": _CLARIFY_SYSTEM_PROMPT,
    "knowledge": _KNOWLEDGE_SYSTEM_PROMPT,
    "verify": _VERIFY_SYSTEM_PROMPT,
}


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_merged_dataset(path: str) -> List[Dict[str, Any]]:
    data = load_json(path)
    samples = []
    for sample_id, item in data.items():
        hist = item.get("history")
        if hist is None and item.get("profile") is not None:
            if isinstance(item["profile"], str):
                hist = item["profile"]
            else:
                hist = "\n\n".join(x["text"] for x in item["profile"][:50])
        samples.append(
            {
                "id": sample_id,
                "question": item["question"],
                "history": hist or "",
                "avg_scores": item["score"]["avg"],
                "sft_sup": item.get("sft_sup", ""),
                "selected_augmentation": item.get("selected_augmentation", ""),
            }
        )
    return samples


def soft_target_from_scores(scores: List[float], beta: float) -> List[float]:
    t = torch.tensor(scores, dtype=torch.float32) / beta
    return torch.softmax(t, dim=0).tolist()


def build_chat_messages_user(history: str, question: str) -> str:
    return QUESTION_USER_PROMPT.format(profile=history, question=question)


@dataclass
class MultiTaskSample:
    task_type: str
    question: str
    history: str
    avg_scores: List[float]
    sft_sup: str
    best_aug: str
    score_target_probs: List[float]


@dataclass
class MultiTaskDataset(Dataset):
    samples: List[MultiTaskSample]
    tokenizer: AutoTokenizer
    max_length: int = 8192

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        user_prompt = build_chat_messages_user(item.history, item.question)

        if item.task_type == "score":
            full_prompt = build_chat_prompt(SCORE_SYSTEM_PROMPT, user_prompt, self.tokenizer)
            target_text = ""
        else:
            sys_prompt = AUG_TO_SYSTEM_PROMPT[item.best_aug]
            full_prompt = build_chat_prompt(sys_prompt, user_prompt, self.tokenizer)
            target_text = item.sft_sup

        full_text = full_prompt + target_text
        tokenized_full = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = tokenized_full["input_ids"][0]
        attention_mask = tokenized_full["attention_mask"][0]
        labels = input_ids.clone()

        tokenized_prompt = self.tokenizer(
            full_prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_len = min(tokenized_prompt["input_ids"].shape[1], len(input_ids))
        labels[:prompt_len] = -100

        if item.task_type == "score":
            score_target = torch.tensor(item.score_target_probs, dtype=torch.float32)
            is_score_task = torch.tensor(1, dtype=torch.int64)
        else:
            score_target = torch.zeros(4, dtype=torch.float32)
            is_score_task = torch.tensor(0, dtype=torch.int64)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "score_target": score_target,
            "is_score_task": is_score_task,
        }


class MultiTaskDataCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, features):
        if not features:
            return {}
        input_ids = []
        attention_mask = []
        labels = []
        valid_features = []
        for f in features:
            if "input_ids" in f:
                input_ids.append(f["input_ids"])
                attention_mask.append(f["attention_mask"])
                labels.append(f["labels"])
                valid_features.append(f)
        if not input_ids:
            return {}
        max_length = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        for ids, mask, label in zip(input_ids, attention_mask, labels):
            pad_length = max_length - len(ids)
            if pad_length > 0:
                padded_input_ids.append(
                    torch.cat([ids, torch.full((pad_length,), self.pad_token_id, dtype=ids.dtype)])
                )
                padded_attention_mask.append(torch.cat([mask, torch.zeros(pad_length, dtype=mask.dtype)]))
                padded_labels.append(torch.cat([label, torch.full((pad_length,), -100, dtype=label.dtype)]))
            else:
                padded_input_ids.append(ids)
                padded_attention_mask.append(mask)
                padded_labels.append(label)
        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
        }
        score_targets = [f.get("score_target", torch.zeros(4)) for f in valid_features]
        is_score_tasks = [f.get("is_score_task", torch.tensor(0, dtype=torch.int64)) for f in valid_features]
        batch["score_target"] = torch.stack(score_targets)
        batch["is_score_task"] = torch.stack(is_score_tasks)
        return batch


class MultiTaskTrainer(Trainer):
    def __init__(self, *args, lambda_qry: float = 1.0, **kwargs):
        self.lambda_qry = lambda_qry
        super().__init__(*args, **kwargs)

    def _remove_unused_columns(self, dataset, description=None):
        return dataset

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        score_target = inputs.pop("score_target")
        is_score_task = inputs.pop("is_score_task")

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
        )
        lm_loss = outputs.loss
        if torch.isnan(lm_loss):
            lm_loss = torch.tensor(0.0, device=outputs.logits.device)

        dev = outputs.logits.device
        act_loss = torch.tensor(0.0, device=dev)
        is_score_mask = is_score_task.to(torch.bool)
        if is_score_mask.any():
            logits = outputs.logits
            attention_mask = inputs["attention_mask"]
            lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(logits.size(0), device=dev)
            last_logits = logits[batch_indices, lengths]
            label_token_ids = torch.tensor(LABEL_TOKEN_IDS, dtype=torch.long, device=dev)
            class_logits = last_logits[:, label_token_ids]
            p_hat = score_target[is_score_mask]
            logits_act = class_logits[is_score_mask]
            log_p_pred = F.log_softmax(logits_act, dim=-1)
            kl_rows = F.kl_div(log_p_pred, p_hat, reduction="none").sum(dim=-1)
            act_loss = kl_rows.mean()

        total_loss = act_loss + self.lambda_qry * lm_loss

        return (total_loss, outputs) if return_outputs else total_loss


def create_lora_model(args):
    global LABEL_TOKEN_IDS
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    LABEL_TOKEN_IDS = []
    for word in LABEL_WORDS:
        ids = tokenizer.encode(word, add_special_tokens=False)
        LABEL_TOKEN_IDS.append(ids[0])

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if getattr(args, "expected_vocab_size", None):
        exp = int(args.expected_vocab_size)
        cur = model.get_input_embeddings().weight.shape[0]
        if cur != exp:
            model.resize_token_embeddings(exp)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def build_multitask_samples(
    all_samples: List[Dict[str, Any]],
    beta: float,
) -> List[MultiTaskSample]:
    multitask_samples: List[MultiTaskSample] = []
    gen_count = 0
    for s in all_samples:
        avg_scores = s["avg_scores"]
        if len(avg_scores) != 4:
            raise ValueError(f"score.avg must have length 4 [PLAN,FACT,REFLECT,ORIGIN], got {len(avg_scores)}")
        probs = soft_target_from_scores(avg_scores, beta)
        multitask_samples.append(
            MultiTaskSample(
                task_type="score",
                question=s["question"],
                history=s["history"],
                avg_scores=avg_scores,
                sft_sup="",
                best_aug="",
                score_target_probs=probs,
            )
        )
        aug = s.get("selected_augmentation") or ""
        sup = s.get("sft_sup") or ""
        if aug != "origin" and sup and aug in AUG_TO_SYSTEM_PROMPT:
            gen_count += 1
            multitask_samples.append(
                MultiTaskSample(
                    task_type="gen",
                    question=s["question"],
                    history=s["history"],
                    avg_scores=avg_scores,
                    sft_sup=sup,
                    best_aug=aug,
                    score_target_probs=probs,
                )
            )
    print(f"[Data] score-task rows: {len(all_samples)}")
    print(f"[Data] gen-task rows   : {gen_count}")
    print(f"[Data] total rows      : {len(multitask_samples)}")
    return multitask_samples


def train_sft(args):
    raw = load_merged_dataset(args.train_file)
    samples = build_multitask_samples(raw, beta=args.beta)
    model, tokenizer = create_lora_model(args)
    dataset = MultiTaskDataset(samples, tokenizer, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_total_limit=3,
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        optim="adamw_torch",
        weight_decay=args.weight_decay,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=MultiTaskDataCollator(tokenizer),
        lambda_qry=args.lambda_qry,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter and tokenizer to {args.output_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Multi-task SFT (metacognitive action + conditional query supervision)")
    p.add_argument("--base_model", type=str, required=True, help="HF id or local path to base causal LM")
    p.add_argument("--train_file", type=str, required=True, help="JSON dict: id -> {question, history, score, ...}")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--lambda_qry",
        type=float,
        default=1.0,
        help="lambda in L_joint = L_act + lambda * L_qry (weights query NLL)",
    )
    p.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="beta in soft target p_hat(a|x) proportional to exp(s(a)/beta)",
    )
    p.add_argument("--max_length", type=int, default=8192)
    p.add_argument("--num_epochs", type=float, default=5.0)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "epoch", "steps"])
    p.add_argument("--bf16", action="store_true", help="Use bf16 (recommended on Ampere+)")
    p.add_argument("--fp16", action="store_true", help="Use fp16 if bf16 unavailable")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--expected_vocab_size",
        type=int,
        default=None,
        help="If set, resize embeddings before LoRA (match checkpoint trained with shrunk vocab)",
    )
    return p.parse_args()


if __name__ == "__main__":
    train_sft(parse_args())

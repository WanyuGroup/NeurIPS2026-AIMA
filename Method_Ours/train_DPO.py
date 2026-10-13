"""
Decomposed DPO (hierarchical preferences, shared contexts within each subset):

  L_DPO = E_{D_act}[ l_dpo(x, a+, a-) ] + E_{D_qry}[ l_dpo((x,a), q+, q-) ]

  D_act: preferences where the chosen trajectory has higher privacy-aware reward R(tau)=U(tau)-S(q)
         and actions differ (a+ != a-). Compared under the shared metacognitive prompt (SCORE_SYSTEM_PROMPT).

  D_qry: preferences with the same action a for both trajectories (same branch system prompt + user x).
         Query completions q+, q- differ; compared under the action-conditioned context.

Reward ordering (R(tau+) > R(tau-)) is assumed to hold when constructing offline JSON pairs.

Input JSON: externally prepared preference samples:
  - strategy_pair: chosen_label, rejected_label, prompt (optional), chosen_aug, ...
  - answer_pair: chosen_answer, rejected_answer, prompt (optional)
  - context: profile, question

Requires: transformers, peft, torch, trl, datasets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer

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

try:
    from trl.trainer.dpo_trainer import DataCollatorForPreference
except Exception:
    from trl import DataCollatorForPreference

from trl import DPOConfig, DPOTrainer

SUBSET_ACT = 0
SUBSET_QRY = 1

AUG_TO_SYSTEM_PROMPT = {
    "clarify": _CLARIFY_SYSTEM_PROMPT,
    "knowledge": _KNOWLEDGE_SYSTEM_PROMPT,
    "verify": _VERIFY_SYSTEM_PROMPT,
}


def load_dpo_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return list(data.values())
    if isinstance(data, list):
        return data
    raise ValueError("DPO JSON must be a list or dict of samples")


def default_user_prompt_from_context(item: Dict[str, Any]) -> str:
    ctx = item.get("context") or {}
    profile = ctx.get("profile", "")
    q = ctx.get("question", "")
    if isinstance(profile, list):
        profile = " ".join(str(x) for x in profile)
    if isinstance(q, list):
        q = " ".join(str(x) for x in q)
    return QUESTION_USER_PROMPT.format(profile=profile or "", question=q or "")


def build_decomposed_dataset(all_items: List[Dict[str, Any]], tokenizer) -> Dataset:
    rows_act: List[Dict[str, Any]] = []
    rows_qry: List[Dict[str, Any]] = []

    for item in all_items:
        sp = item.get("strategy_pair") or {}
        ap = item.get("answer_pair") or {}

        user_prompt = (sp.get("prompt") or "").strip()
        if not user_prompt:
            user_prompt = default_user_prompt_from_context(item)

        chosen_strat = (sp.get("chosen_label") or "").strip()
        rejected_strat = (sp.get("rejected_label") or "").strip()

        if chosen_strat and rejected_strat and chosen_strat != rejected_strat:
            full_prompt = build_chat_prompt(SCORE_SYSTEM_PROMPT, user_prompt, tokenizer)
            rows_act.append(
                {
                    "prompt": full_prompt,
                    "chosen": chosen_strat,
                    "rejected": rejected_strat,
                    "subset_id": SUBSET_ACT,
                }
            )

        if ap and ap.get("chosen_answer") and ap.get("rejected_answer"):
            aug_type = sp.get("chosen_aug")
            if aug_type in AUG_TO_SYSTEM_PROMPT:
                sys_prompt = AUG_TO_SYSTEM_PROMPT[aug_type]
                up = (ap.get("prompt") or "").strip() or user_prompt
                chosen_ans = ap.get("chosen_answer", "")
                rejected_ans = ap.get("rejected_answer", "")
                if up and chosen_ans and rejected_ans and chosen_ans != rejected_ans:
                    full_prompt = build_chat_prompt(sys_prompt, up, tokenizer)
                    rows_qry.append(
                        {
                            "prompt": full_prompt,
                            "chosen": chosen_ans,
                            "rejected": rejected_ans,
                            "subset_id": SUBSET_QRY,
                        }
                    )

    rows = rows_act + rows_qry
    stats = {"D_act": len(rows_act), "D_qry": len(rows_qry), "total": len(rows)}
    print(f"[Decomposed DPO dataset] {stats}")
    if not rows:
        return Dataset.from_list([])
    ds = Dataset.from_list(rows)
    return ds.shuffle(seed=42)


def _gather_dpo_subbatch(inputs: Dict[str, Any], idx: torch.Tensor) -> Dict[str, Any]:
    if idx.numel() == 0:
        return {}
    device = idx.device
    idx = idx.to(dtype=torch.long, device=device)
    n_seq = inputs["input_ids"].shape[0]
    half = n_seq // 2
    rows = torch.cat([idx, idx + half], dim=0)
    out: Dict[str, Any] = {}
    for k, v in inputs.items():
        if k == "subset_id":
            continue
        if isinstance(v, torch.Tensor):
            if v.shape[0] == n_seq:
                out[k] = v[rows]
            elif v.shape[0] == half:
                out[k] = v[idx]
            else:
                out[k] = v
        else:
            out[k] = v
    return out


class DecomposedDPOTrainer(DPOTrainer):
    def _prepare_dataset(
        self,
        dataset,
        processing_class,
        args,
        dataset_name: str,
    ):
        subset_col: Optional[List[int]] = None
        if hasattr(dataset, "column_names") and dataset.column_names and "subset_id" in dataset.column_names:
            subset_col = list(dataset["subset_id"])
        out = super()._prepare_dataset(dataset, processing_class, args, dataset_name)
        if subset_col is not None and hasattr(out, "column_names"):
            if hasattr(out, "__len__") and len(out) == len(subset_col):
                out = out.add_column("subset_id", subset_col)
            else:
                print(
                    f"[warn] subset_id length {len(subset_col)} != tokenized dataset {len(out)}; "
                    "skipping subset_id reattach"
                )
        return out

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("return_outputs=True is not supported for DecomposedDPOTrainer.compute_loss")
        sid = inputs.pop("subset_id", None)
        if sid is None:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=False,
                num_items_in_batch=num_items_in_batch,
            )

        half = inputs["input_ids"].shape[0] // 2
        if sid.shape[0] != half:
            raise ValueError(f"subset_id batch {sid.shape[0]} != comparisons {half}")

        act_ix = (sid == SUBSET_ACT).nonzero(as_tuple=True)[0]
        qry_ix = (sid == SUBSET_QRY).nonzero(as_tuple=True)[0]

        total: Optional[torch.Tensor] = None
        for ix in (act_ix, qry_ix):
            if ix.numel() == 0:
                continue
            sub = _gather_dpo_subbatch(inputs, ix)
            li = self._compute_loss(model, sub, return_outputs=False)
            total = li if total is None else total + li

        if total is None:
            p0 = next(model.parameters())
            return p0.sum() * 0.0
        return total


class CollatorWithSubset:
    def __init__(
        self,
        pad_token_id: int,
        max_length: Optional[int],
        truncation_mode: str = "keep_start",
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.inner = DataCollatorForPreference(
            pad_token_id=pad_token_id,
            max_length=max_length,
            truncation_mode=truncation_mode,
            pad_to_multiple_of=pad_to_multiple_of,
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        subset_ids = [int(f["subset_id"]) for f in features]
        batch = self.inner(features)
        batch["subset_id"] = torch.tensor(subset_ids, dtype=torch.long)
        return batch


def load_policy_model(base_model: str, adapter_path: str, use_lora: bool = True):
    adapter_cfg = Path(adapter_path) / "adapter_config.json"
    if adapter_cfg.is_file():
        print(f"[LoRA] Loading base={base_model}, adapter={adapter_path}")
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if use_lora:
            base_model_obj = prepare_model_for_kbit_training(base_model_obj)
        policy_model = PeftModel.from_pretrained(base_model_obj, adapter_path, is_trainable=True)
        policy_model.print_trainable_parameters()
        return policy_model

    print(f"[Model] Loading full weights from {adapter_path}")
    model = AutoModelForCausalLM.from_pretrained(
        adapter_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if use_lora:
        model = prepare_model_for_kbit_training(model)
    return model


def train_dpo(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    items = load_dpo_items(args.dpo_file)
    raw_ds = build_decomposed_dataset(items, tokenizer)
    if len(raw_ds) == 0:
        print("No valid pairs; exiting.")
        return

    split = raw_ds.train_test_split(test_size=args.eval_split, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    policy_model = load_policy_model(args.base_model, args.adapter_path, use_lora=not args.no_lora)

    peft_config = None
    if args.no_lora:
        peft_config = None
    elif not isinstance(policy_model, PeftModel):
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        optim="adamw_torch",
        report_to="none",
    )

    collator = CollatorWithSubset(
        pad_token_id=int(tokenizer.pad_token_id),
        max_length=args.max_length,
        truncation_mode="keep_start",
    )

    trainer_kwargs = dict(
        model=policy_model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        data_collator=collator,
    )

    try:
        trainer = DecomposedDPOTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = DecomposedDPOTrainer(tokenizer=tokenizer, **trainer_kwargs)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decomposed hierarchical DPO (preference optimization)")
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--adapter_path", type=str, required=True, help="SFT LoRA dir or full model path")
    p.add_argument("--dpo_file", type=str, required=True, help="JSON list or dict of DPO items")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--beta", type=float, default=0.1, help="DPO beta (TRL)")
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--max_prompt_length", type=int, default=1800)
    p.add_argument("--num_epochs", type=float, default=4.0)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--eval_split", type=float, default=0.05)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_strategy", type=str, default="steps", choices=["no", "epoch", "steps"])
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--eval_strategy", type=str, default="steps", choices=["no", "epoch", "steps"])
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--no_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    return p.parse_args()


if __name__ == "__main__":
    train_dpo(parse_args())

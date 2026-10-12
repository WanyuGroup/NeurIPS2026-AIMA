import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import os
from typing import Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.formetters import (
    QUESTION_USER_PROMPT,
    SCORE_SYSTEM_PROMPT,
    _CLARIFY_SYSTEM_PROMPT,
    _KNOWLEDGE_SYSTEM_PROMPT,
    _VERIFY_SYSTEM_PROMPT,
)
from utils.custom_llm import build_chat_prompt, load_json, save_json
from utils.privacy_gate import (
    build_privacy_feedback_en,
    compute_privacy_score,
    extract_sensitive_entities,
    get_embedder,
    pri_rewrite_user_block,
)

LABEL_TO_AUG = {
    "PLAN": "clarify",
    "FACT": "knowledge",
    "REFLECT": "verify",
    "ORIGIN": "origin",
}
AUG_TO_SYSTEM_PROMPT = {
    "clarify": _CLARIFY_SYSTEM_PROMPT,
    "knowledge": _KNOWLEDGE_SYSTEM_PROMPT,
    "verify": _VERIFY_SYSTEM_PROMPT,
}
LABEL_WORDS = ["PLAN", "FACT", "REFLECT", "ORIGIN"]
LABEL_TOKEN_IDS: List[int] = []

AUG_LABEL_EN = {
    "clarify": "PLAN — clarified / disambiguated query for cloud use",
    "knowledge": "FACT — general-knowledge retrieval query",
    "verify": "REFLECT — privacy-preserving process description",
}

parser = argparse.ArgumentParser(description="Run Strategy Prediction and Augmentation with LoRA")
parser.add_argument(
    "--model_addr",
    type=str,
    default="Qwen/Qwen2.5-7B-Instruct",
    help="HF model id or local folder for the causal LM weights",
)
parser.add_argument("--lora_addr", type=str, default=None)
parser.add_argument("--input_addr", type=str, default="./data/retriever_rank/")
parser.add_argument("--output_addr", type=str, default="./outputs/step1/")
parser.add_argument(
    "--dataset_names",
    nargs="+",
    default=["dataset_eval_A"],
    help="Logical dataset id used in artifact filenames (override for your benchmark)",
)
parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation")
parser.add_argument(
    "--model_name",
    type=str,
    default="Qwen2.5-7B-Instruct",
    help="Tag embedded in output filenames (default matches Qwen2.5-7B-Instruct)",
)
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--num_contexts", type=int, default=10)
parser.add_argument("--stage", type=str, default="SFT_DPO")
parser.add_argument(
    "--no_privacy_rewrite",
    action="store_true",
    help="Disable privacy-constrained iterative rewriting for refined outputs",
)
parser.add_argument("--k_max_rewrite", type=int, default=5, help="Max rewrite rounds including initial draft")
parser.add_argument("--delta_sem", type=float, default=0.65, help="Cosine threshold for semantic leakage detector")
parser.add_argument(
    "--st_model",
    type=str,
    default="sentence-transformers/all-MiniLM-L6-v2",
    help="SentenceTransformer id for SEM component; empty disables neural embeddings (lexical SEM fallback only)",
)
parser.add_argument(
    "--no_semantic_embedder",
    action="store_true",
    help="Do not load sentence-transformers even if --st_model is set",
)
parser.add_argument(
    "--expected_vocab_size",
    type=int,
    default=None,
    help="If set with --lora_addr, resize embeddings before loading adapter when sizes mismatch",
)


def load_model_and_tokenizer(
    base_model_path: str,
    lora_path: str,
    expected_vocab_size: Optional[int] = None,
):
    global LABEL_TOKEN_IDS

    print(f"Loading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    LABEL_TOKEN_IDS = []
    print("[Model] Mapping labels to natural language tokens:")
    for word in LABEL_WORDS:
        ids = tokenizer.encode(word, add_special_tokens=False)
        target_id = ids[0]
        LABEL_TOKEN_IDS.append(target_id)
        print(f"  Word: '{word}' -> Token ID: {target_id}")

    print(f"Loading base model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if lora_path is not None and expected_vocab_size is not None:
        cur = model.get_input_embeddings().weight.shape[0]
        if cur != expected_vocab_size:
            print(
                f"[Fix] Resizing token embeddings from {cur} to {expected_vocab_size} to match LoRA checkpoint."
            )
            model.resize_token_embeddings(expected_vocab_size)

    if lora_path is not None:
        print(f"Loading LoRA weights from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    return model, tokenizer


def predict_strategy(model, tokenizer, history: str, question: str) -> str:
    user_prompt = QUESTION_USER_PROMPT.format(profile=history, question=question)
    full_prompt = build_chat_prompt(SCORE_SYSTEM_PROMPT, user_prompt, tokenizer)

    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[:, -1, :]
        target_token_tensor = torch.tensor(LABEL_TOKEN_IDS, device=model.device)
        label_logits = next_token_logits[0, target_token_tensor]

        best_idx = torch.argmax(label_logits).item()
        predicted_label = LABEL_WORDS[best_idx]
        best_aug = LABEL_TO_AUG[predicted_label]

    return best_aug


def _run_sl_generation(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt_text: str,
    args,
) -> str:
    full_prompt = build_chat_prompt(system_prompt, user_prompt_text, tokenizer)
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            temperature=args.temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def _resolve_embedder(args):
    if getattr(args, "no_semantic_embedder", False):
        return None
    mid = (getattr(args, "st_model", None) or "").strip()
    if not mid:
        return None
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("[privacy] sentence-transformers not installed; SEM uses lexical fallback only.")
        return None
    cache_key = "_privacy_embedder"
    emb = getattr(args, cache_key, None)
    if emb is None:
        emb = get_embedder(mid)
        setattr(args, cache_key, emb)
    return emb


def generate_aug_content(
    model,
    tokenizer,
    aug_type: str,
    history: str,
    question: str,
    args,
) -> Tuple[str, Optional[Dict]]:
    if aug_type == "origin":
        return "", None

    system_prompt = AUG_TO_SYSTEM_PROMPT[aug_type]
    base_user = QUESTION_USER_PROMPT.format(profile=history, question=question)

    if getattr(args, "no_privacy_rewrite", False):
        out = _run_sl_generation(model, tokenizer, system_prompt, base_user, args)
        return out, None

    entities = extract_sensitive_entities(history)
    embedder = _resolve_embedder(args)
    k_max = max(1, int(args.k_max_rewrite))
    aug_en = AUG_LABEL_EN.get(aug_type, aug_type)

    candidates: List[Tuple[str, float]] = []
    candidate = _run_sl_generation(model, tokenizer, system_prompt, base_user, args)

    for k in range(k_max):
        detail = compute_privacy_score(candidate, entities, embedder, float(args.delta_sem))
        candidates.append((candidate, detail.total))
        if detail.total == 0.0:
            meta = {
                "privacy_score_final": 0.0,
                "rewrite_rounds_used": k + 1,
                "converged": True,
                "s_pii": detail.s_pii,
                "s_ner": detail.s_ner,
                "s_sem": detail.s_sem,
            }
            return candidate, meta
        if k + 1 >= k_max:
            break
        fb = build_privacy_feedback_en(detail)
        rw_user = base_user + pri_rewrite_user_block(k + 1, fb, candidate, aug_en)
        candidate = _run_sl_generation(model, tokenizer, system_prompt, rw_user, args)

    best_text, best_s = min(candidates, key=lambda x: x[1])
    final_detail = compute_privacy_score(best_text, entities, embedder, float(args.delta_sem))
    meta = {
        "privacy_score_final": final_detail.total,
        "rewrite_rounds_used": len(candidates),
        "converged": final_detail.total == 0.0,
        "s_pii": final_detail.s_pii,
        "s_ner": final_detail.s_ner,
        "s_sem": final_detail.s_sem,
        "selected_lowest_score_among_candidates": True,
    }
    return best_text, meta


def main_test(dataset_name: str, model, tokenizer, args, is_proprietary_llm=False):
    print(f"\nTesting on dataset: {dataset_name}")

    rank_json = f"{args.model_name}_{dataset_name}_{args.split}_rank_filter_100.json"
    candidate = os.path.join(args.input_addr, rank_json)
    if os.path.isfile(candidate):
        data_path = candidate
    elif os.path.isfile(args.input_addr):
        data_path = args.input_addr
    else:
        data_path = candidate
    dataset_history = None
    if not os.path.exists(data_path):
        print(f"Skipping {dataset_name}: File not found.")
        return None

    print("Loading test data...")
    dataset_question = load_json(data_path)
    print(f"Processing {len(dataset_question)} samples...")

    aug_counts = {"clarify": 0, "knowledge": 0, "verify": 0, "origin": 0}
    results = {}
    for q_item in tqdm(dataset_question, desc="Processing Samples"):
        sample_id = q_item["id"]
        question = q_item["question"]

        if dataset_history is not None and sample_id not in dataset_history:
            continue

        history = "\n\n".join([x["text"] for x in q_item["profile"][: args.num_contexts]])

        best_aug = predict_strategy(model, tokenizer, history, question)
        aug_counts[best_aug] += 1

        output, priv_meta = generate_aug_content(model, tokenizer, best_aug, history, question, args)

        row = {
            "question": question,
            "output": output,
            "best_aug": best_aug,
        }
        if priv_meta is not None:
            row["privacy_gate"] = priv_meta
        results[sample_id] = row

    output_filename = f"{args.model_name}_{dataset_name}_{args.split}_{args.stage}_output.json"
    output_path = os.path.join(args.output_addr, output_filename)
    save_json(results, output_path)
    print(f"Results saved to: {output_path}")

    total_samples = len(results)
    aug_stats = {
        "total_samples": total_samples,
        "augmentation_stats": {
            aug: {
                "count": count,
                "percentage": round(count / total_samples * 100, 2) if total_samples > 0 else 0,
            }
            for aug, count in aug_counts.items()
        },
    }

    stats_filename = f"{args.model_name}_{dataset_name}_{args.split}_stats.json"
    stats_path = os.path.join(args.output_addr, stats_filename)
    save_json(aug_stats, stats_path)

    print(f"--- Stats for {dataset_name} ---")
    for aug, count in aug_counts.items():
        percentage = round(count / total_samples * 100, 2) if total_samples > 0 else 0
        print(f"{aug}: {count} ({percentage}%)")

    return aug_stats


if __name__ == "__main__":
    args = parser.parse_args()

    print("Arguments Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    model, tokenizer = load_model_and_tokenizer(
        args.model_addr,
        args.lora_addr,
        expected_vocab_size=args.expected_vocab_size,
    )

    all_stats = {}
    for dataset in args.dataset_names:
        stats = main_test(dataset, model, tokenizer, args)
        if stats:
            all_stats[dataset] = stats

    print(f"\n{'=' * 60}")
    print("OVERALL STATISTICS ACROSS ALL DATASETS")
    print("=" * 60)

    for dataset, stats in all_stats.items():
        print(f"\nDataset: {dataset}")
        for aug, info in stats["augmentation_stats"].items():
            print(f"  {aug}: {info['count']} ({info['percentage']}%)")

# AIMA

Official code repository for **Learning to Ask: Metacognitive Action Policy for Large and Small Language Model Collaboration**.

**Accepted at NeurIPS 2026.**

Jiayuan Zhang, Jianwei Niu, Xuefeng Liu, Haotian Yang, Guogang Zhu, Yuhui Niu, Wanyu Lin*, and Xinghao Wu.

*Corresponding author.

In Proceedings of the 40th Annual Conference on Neural Information Processing Systems (NeurIPS), Sydney, Australia, December 2026, pp. 1–33.

## Installation and usage

Run from the package root (containing `Method_Ours/`, `data/`, `utils/`). Install: `pip install -r requirements.txt`.

## Training

1. **Joint SFT** — `Method_Ours/train_SFT_multitask.py` (see file header for loss).  
   Example: `python Method_Ours/train_SFT_multitask.py --base_model <id or path> --train_file <json> --output_dir <dir> --bf16`

2. **Decomposed DPO** — `Method_Ours/train_DPO.py` (after SFT).  
   Example: `python Method_Ours/train_DPO.py --base_model <id> --adapter_path <lora or model> --dpo_file <json> --output_dir <dir>`

## Inference (three scripts)

| Step | Script | Role |
|------|--------|------|
| 1 | `Ours1_askquestion.py` | Local LM + optional LoRA; strategy + branch outputs; optional privacy rewrite (`privacy_gate.py`). |
| 2 | `Ours2_LLMgeneration.py` | Chat API per branch (configure `OPENAI_API_KEY`, `OPENAI_BASE_URL`; defaults use placeholder `xxxxx` until set). |
| 3 | `Ours3_SLMgenerate.py` | vLLM (default) or `--openai` for final answers from retrieval prompts. |

Defaults align filenames across steps: base weights `Qwen/Qwen2.5-7B-Instruct`, file tag `Qwen2.5-7B-Instruct`, `--file_model_tag` in Step 2 matches Step 1 `--model_name`. Do not commit real API keys.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{zhang2026learningtoask,
  title     = {Learning to Ask: Metacognitive Action Policy for Large and Small Language Model Collaboration},
  author    = {Zhang, Jiayuan and Niu, Jianwei and Liu, Xuefeng and Yang, Haotian and Zhu, Guogang and Niu, Yuhui and Lin, Wanyu and Wu, Xinghao},
  booktitle = {Proceedings of the 40th Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  month     = dec,
  address   = {Sydney, Australia},
  pages     = {1--33},
  url       = {https://github.com/ZhangJiayuan-BUAA/AIMA}
}
```

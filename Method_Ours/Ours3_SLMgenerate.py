import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import os

import srsly
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from data.formetters import get_answer_formatter, prepare_prompts_dict
from utils.custom_llm import OpenAILLM
from utils.json_utils import str_to_json


def load_llm(model_addr, cache_dir, num_gpus: int = 1):
    adapter_config_path = os.path.join(model_addr, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        return LLM(model_addr, download_dir=cache_dir, tensor_parallel_size=num_gpus, enforce_eager=True), None
    with open(adapter_config_path, "r") as f:
        config = json.load(f)
        base_model = config.get("base_model_name_or_path")
    llm = LLM(base_model, download_dir=cache_dir, enable_lora=True, tensor_parallel_size=num_gpus)
    lora = LoRARequest("custom_lora", 1, lora_path=model_addr)
    return llm, lora


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_addr",
    type=str,
    default="Qwen/Qwen2.5-7B-Instruct",
    help="vLLM: HF model path. Ignored when --openai (use --openai_model).",
)
parser.add_argument(
    "--openai_model",
    type=str,
    default="gpt-4o",
    help="When --openai: Chat Completions model id (default gpt-4o)",
)
parser.add_argument(
    "--inputs_addr",
    type=str,
    default="./outputs/step2_cloud.json",
    help="Typically the Step 2 JSON; override to your artifact path",
)
parser.add_argument(
    "--inputs_profile",
    type=str,
    default="./data/retriever_rank/profile_rank.json",
)
parser.add_argument(
    "--output_addr",
    type=str,
    default="./outputs/step3_SLMresponse.json",
)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--max_tokens", type=int, default=4096)
parser.add_argument("--num_generated_outputs", type=int, default=1)
parser.add_argument("--num_contexts", type=int, default=10)
parser.add_argument("--max_retries", type=int, default=20)
parser.add_argument("--openai", action="store_true")
parser.add_argument("--api_key_addr", type=str, default="", help="Optional file containing API key (one line)")
parser.add_argument("--cache_dir", default="./cache")
parser.add_argument("--num_gpus", type=int, default=1)

if __name__ == "__main__":
    args = parser.parse_args()
    dataset = srsly.read_json(args.inputs_addr)
    dataset_profile = srsly.read_json(args.inputs_profile)

    if args.openai:
        api_key = os.environ.get("OPENAI_API_KEY")
        if args.api_key_addr and os.path.isfile(args.api_key_addr):
            with open(args.api_key_addr, "r", encoding="utf-8") as file:
                api_key = file.read().strip()
        if not api_key or api_key == "xxxxx":
            raise RuntimeError(
                "Set OPENAI_API_KEY or pass --api_key_addr with a real key file when using --openai."
            )
        is_proprietary_llm = True
        llm = OpenAILLM(model_name=args.openai_model, api_key=api_key)
        tokenizer = None
    else:
        llm, lora_req = load_llm(args.model_addr, args.cache_dir, args.num_gpus)
        tokenizer = llm.get_tokenizer()
        is_proprietary_llm = False

    formater = get_answer_formatter(tokenizer, args.num_contexts, proprietary_llm=is_proprietary_llm)
    prompts, ids = prepare_prompts_dict(dataset, dataset_profile, formater)

    outputs_dict = {}
    temperature = args.temperature
    retries = 0
    while prompts:
        retries += 1
        wrongs = []
        sampling_params = SamplingParams(
            temperature=temperature, top_p=args.top_p, max_tokens=args.max_tokens, logprobs=1
        )
        if not args.openai:
            print(lora_req)
            outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)
        else:
            outputs = llm.generate(prompts, sampling_params)

        for id, prompt, output in zip(ids, prompts, outputs):
            if id not in outputs_dict:
                outputs_dict[id] = []
            try:
                text_output = output.outputs[0].text
                response_obj = str_to_json(text_output)
                outputs_dict[id].append(
                    {
                        "prompt": prompt,
                        "whole_output": response_obj,
                        "output": response_obj["personalized_answer"],
                        "log_prob": output.outputs[0].cumulative_logprob,
                    }
                )

            except Exception as e:
                if retries > args.max_retries:
                    outputs_dict[id].append(
                        {
                            "prompt": prompt,
                            "whole_output": "",
                            "output": "",
                            "log_prob": 0,
                        }
                    )
                    continue
                if temperature < 1:
                    temperature += 0.1
                print(e)
                wrongs.append((id, prompt))
        prompts = []
        ids = []
        for wrong in wrongs:
            ids.append(wrong[0])
            prompts.append(wrong[1])
    os.makedirs(os.path.dirname(args.output_addr) or ".", exist_ok=True)
    with open(args.output_addr, "w") as file:
        json.dump(outputs_dict, file, indent=4)

import concurrent.futures as futures
import json
import os
from typing import Any, Dict, List, Union

import backoff
import tqdm
from openai import OpenAI, RateLimitError


def _to_chat_messages(prompt: Union[str, List[Dict]]) -> List[Dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    out: List[Dict[str, str]] = []
    for m in prompt:
        if isinstance(m, dict) and "role" in m and "content" in m:
            out.append({"role": str(m["role"]), "content": str(m["content"])})
    return out if out else [{"role": "user", "content": str(prompt)}]


def batchify(lst, n):
    return [lst[i : i + n] for i in range(0, len(lst), n)]


class TextHelper:
    def __init__(self, text):
        self.text = text
        self.cumulative_logprob = 0


class OutputHelper:
    def __init__(self, output):
        self.outputs = [TextHelper(output)]


@backoff.on_exception(backoff.expo, RateLimitError)
def get_completion_from_gpt(messages, model_name, max_tokens, api_key, temperature):
    base_url = os.environ.get("OPENAI_BASE_URL") or None
    effective_key = api_key if api_key else os.environ.get("OPENAI_API_KEY")
    if not effective_key or effective_key == "xxxxx":
        raise RuntimeError("Set OPENAI_API_KEY before making cloud requests.")
    client = OpenAI(base_url=base_url, api_key=effective_key)
    msg_list = _to_chat_messages(messages)
    response = client.chat.completions.create(
        messages=msg_list,
        model=model_name,
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )
    return response.choices[0].message.content


class OpenAILLM:

    def __init__(self, model_name, api_key) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.n_threads = 8

    def generate(self, prompts, sampling_param):
        results = []
        barches = batchify(prompts, self.n_threads)
        for batch in tqdm.tqdm(barches):
            temp_results = []
            with futures.ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                for data in batch:
                    temp_results.append(
                        executor.submit(
                            get_completion_from_gpt,
                            data,
                            self.model_name,
                            sampling_param.max_tokens,
                            self.api_key,
                            sampling_param.temperature,
                        )
                    )
            for temp_result in temp_results:
                results.append(OutputHelper(temp_result.result()))
        return results


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_chat_prompt(system_prompt: str, user_prompt: str, tokenizer) -> str:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return full_prompt

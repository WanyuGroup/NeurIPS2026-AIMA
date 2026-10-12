#!/usr/bin/env python

import os
import random
import time

from openai import OpenAI


class LLMs:
    def __init__(
        self,
        model="gpt-4o",
        request_type="openai",
        parameters=None,
        max_retries=3,
    ):
        self.model = model
        self.request_type = request_type
        self.parameters = parameters if parameters is not None else {"temperature": 0.9, "top_p": 0.7}
        self.max_retries = max_retries

        api_key = os.getenv("OPENAI_API_KEY", "xxxxx")
        base_url = os.getenv("OPENAI_BASE_URL", "xxxxx")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def request(self, prompt):
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    **self.parameters,
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                msg = str(e)
                if "401" in msg or "Unauthorized" in msg:
                    print("401 Unauthorized. Check OPENAI_API_KEY, OPENAI_BASE_URL, or model access.")
                    print(msg)
                    break
                elif any(x in msg for x in ["429", "502", "503", "504"]):
                    sleep_time = 2**attempt + random.random()
                    print(f"Transient error, retry in {sleep_time:.1f}s: {msg}")
                    time.sleep(sleep_time)
                    continue
                else:
                    print(f"Error: {msg}")
                    break
        return None

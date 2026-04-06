import os
import time


class LLMClient:
    def __init__(
        self,
        model=None,
        api_key=None,
        base_url=None,
        max_retries=3,
        retry_sleep=5,
    ):
        self.model = model or os.getenv("OPENAI_MODEL")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep

        if not self.api_key:
            raise ValueError("Missing OPENAI_API_KEY. Put it in the project root .env file.")

        from openai import OpenAI

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = OpenAI(**client_kwargs)

    def chat(self, prompt, temperature=0.0, max_tokens=4096):
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep)
        raise RuntimeError(f"LLM request failed after {self.max_retries} attempts: {last_error}")

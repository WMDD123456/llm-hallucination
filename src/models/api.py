from openai import OpenAI
from src.config import get_model_config, SYSTEM_PROMPT, TEMPERATURE, MAX_TOKENS


class ModelAPI:
    """统一的模型调用接口，支持 DeepSeek / GLM"""

    def __init__(self, model_name: str):
        cfg = get_model_config(model_name)
        self.model_name = model_name
        self.display_name = cfg["display_name"]
        self.client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content or ""

    def __repr__(self):
        return f"ModelAPI({self.display_name})"

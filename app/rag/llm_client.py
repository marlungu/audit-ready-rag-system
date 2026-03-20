import time
import logging

import boto3

from app.config import settings

logger = logging.getLogger(__name__)


class BedrockClaudeClient:
    def __init__(self):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
        )
        self.model_id = settings.chat_model_id

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float | None = None,) -> str:
        if temperature is None:
            temperature = settings.temperature

        last_error = None

        for attempt in range(3):
            try:
                response = self.client.converse(
                    modelId=self.model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": [{"text": prompt}],
                        }
                    ],
                    inferenceConfig={
                        "maxTokens": max_tokens,
                        "temperature": temperature,
                    },
                )

                content = response["output"]["message"]["content"]
                text = "".join(part.get("text", "") for part in content)

                usage = response.get("usage", {})
                logger.debug(
                    "Bedrock generation: model=%s input_tokens=%s output_tokens=%s",
                    self.model_id,
                    usage.get("inputTokens"),
                    usage.get("outputTokens"),
                )

                return text
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Bedrock generate attempt %s failed: %s",
                    attempt + 1,
                    exc,
                )
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))

        raise last_error
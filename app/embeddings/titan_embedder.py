import json
import logging

import boto3

from app.config import settings

logger = logging.getLogger(__name__)


class TitanEmbedder:

    def __init__(self):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
        )

        self.model_id = settings.embedding_model_id
        self.dimensions = settings.embedding_dimensions
        self.normalize = settings.embedding_normalize

    def embed_text(self, text: str) -> list[float]:
        """Generate a normalized embedding vector using Amazon Titan V2.

        Explicitly passes dimensions and normalization to avoid relying
        on model defaults.  Normalized vectors make cosine similarity
        equivalent to dot product, which is faster to compute.
        """

        body = {
            "inputText": text,
            "dimensions": self.dimensions,
            "normalize": self.normalize,
        }

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        response_body = json.loads(response["body"].read())

        return response_body["embedding"]
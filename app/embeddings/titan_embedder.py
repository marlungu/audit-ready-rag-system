import json
import boto3
from app.config import settings


class TitanEmbedder:

    def __init__(self):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )

        self.model_id = "amazon.titan-embed-text-v2:0"

    def embed_text(self, text: str):

        body = {
            "inputText": text
        }

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        response_body = json.loads(response["body"].read())

        return response_body["embedding"]
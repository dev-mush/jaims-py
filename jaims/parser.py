from typing import List
from jaims.entities import Message, LLMParams, Config
import yaml


class JAIMSParser:
    model: str
    provider: str
    config: Config
    llm_params: LLMParams
    messages: List[Message]

    def __init__(self, document: str):
        try:
            self.parsed = yaml.safe_load(document)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML: {e}")
        if not self.parsed:
            raise ValueError("Document is empty")

        if "model" not in self.parsed:
            raise ValueError("Model not provided")

        self.model = self.parsed["model"]

        if "provider" not in self.parsed:
            raise ValueError("Provider not provided")

        self.provider = self.parsed["provider"]
        if "config" not in self.parsed:
            self.config = Config()
        else:
            self.config = Config.model_validate(self.parsed["config"])

        if "llm_params" not in self.parsed:
            self.llm_params = LLMParams()
        else:
            self.llm_params = LLMParams.model_validate(self.parsed["llm_params"])

        if "messages" not in self.parsed:
            self.messages = []
        else:
            self.messages = self.messages = [
                Message.model_validate(message) for message in self.parsed["messages"]
            ]

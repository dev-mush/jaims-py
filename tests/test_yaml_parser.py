from typing import List, Optional
import unittest
from jaims.entities import (
    Config,
    LLMParams,
    Message,
    MessageRole,
    ContentType,
    ImageContent,
)
from jaims.parser import JAIMSParser
from pydantic import BaseModel, Field


mock_document = """
model: "gpt-4o"
provider: "openai"
llm_params:
  temperature: 0.7
  max_tokens: 4000
  response_format:
    some: "value"
config:
  max_retries: 15
  retry_delay: 10
  exponential_base: 2
  exponential_delay: 1
  exponential_cap: null
  jitter: true
  platform_specific_options: 
    some: "value"
"""

messages = """
messages:
  - role: "user"
    contents:
      - "Hello, I'm Jacob"
      - "What's your name?"
    name: "Jacob"
  - role: "assistant"
    contents:
      - "Hello Jacob, I'm Jaims"
      - "How can I help you today?"
"""


class TestYAMLParser(unittest.TestCase):

    def test_parser_raises_when_model_not_provided(self):
        doc = ""
        with self.assertRaises(ValueError) as e:
            parser = JAIMSParser(doc)

    def test_parser_raises_when_provider_not_provided(self):
        doc = "model: 'gpt-4o'"
        with self.assertRaises(ValueError) as e:
            parser = JAIMSParser(doc)

    def test_parser_returns_parsed_provider_and_model(self):
        parser = JAIMSParser(mock_document)
        self.assertEqual(parser.model, "gpt-4o")
        self.assertEqual(parser.provider, "openai")

    def test_parser_returns_default_config_when_missing(self):
        doc = "model: 'gpt-4o'\nprovider: 'openai'"
        parser = JAIMSParser(doc)

        # assert is of class Config and not None
        self.assertIsInstance(parser.config, Config)

    def test_parser_returns_parsed_config(self):
        parser = JAIMSParser(mock_document)
        self.assertEqual(parser.config.max_retries, 15)
        self.assertEqual(parser.config.retry_delay, 10)
        self.assertEqual(parser.config.exponential_base, 2)
        self.assertEqual(parser.config.exponential_delay, 1)
        self.assertIsNone(parser.config.exponential_cap)
        self.assertTrue(parser.config.jitter)
        self.assertEqual(parser.config.platform_specific_options, {"some": "value"})

    def test_parser_returns_default_llm_params_when_missing(self):
        doc = "model: 'gpt-4o'\nprovider: 'openai'"
        parser = JAIMSParser(doc)

        self.assertIsInstance(parser.llm_params, LLMParams)

    def test_parser_returns_parsed_llm_params(self):
        parser = JAIMSParser(mock_document)
        self.assertEqual(parser.llm_params.temperature, 0.7)
        self.assertEqual(parser.llm_params.max_tokens, 4000)
        self.assertEqual(parser.llm_params.response_format, {"some": "value"})

    def test_parser_returns_empty_messages_when_missing(self):
        doc = "model: 'gpt-4o'\nprovider: 'openai'"
        parser = JAIMSParser(doc)

        self.assertIsInstance(parser.messages, List)
        self.assertEqual(len(parser.messages), 0)

    def test_parser_returns_parsed_messages(self):
        doc = mock_document + messages
        parser = JAIMSParser(doc)

        expected_messages = [
            Message(
                role=MessageRole.USER,
                contents=["Hello, I'm Jacob", "What's your name?"],
                name="Jacob",
            ),
            Message(
                role=MessageRole.ASSISTANT,
                contents=["Hello Jacob, I'm Jaims", "How can I help you today?"],
            ),
        ]

        self.assertEqual(parser.messages, expected_messages)


if __name__ == "__main__":
    unittest.main()

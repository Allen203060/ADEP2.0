import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.agent_config import AgentConfig
from google.adk.models.lite_llm import LiteLlm

class TestAgentConfig(unittest.TestCase):
    def test_default_google_config(self):
        with patch.dict(os.environ, {"ADEP_PROVIDER": "google", "ADEP_MODEL_NAME": "gemini-3.1-flash-lite"}):
            config = AgentConfig()
            self.assertEqual(config.PROVIDER, "google")
            self.assertEqual(config.MODEL_NAME, "gemini-3.1-flash-lite")
            self.assertEqual(config.MODEL, "gemini-3.1-flash-lite")

    def test_ollama_config(self):
        with patch.dict(os.environ, {"ADEP_PROVIDER": "ollama", "ADEP_MODEL_NAME": "gemma3:latest"}):
            config = AgentConfig()
            self.assertEqual(config.PROVIDER, "ollama")
            self.assertEqual(config.MODEL_NAME, "gemma3:latest")
            self.assertTrue(isinstance(config.MODEL, LiteLlm))
            self.assertEqual(config.MODEL.model, "ollama_chat/gemma3:latest")

    def test_openrouter_config(self):
        with patch.dict(os.environ, {"ADEP_PROVIDER": "openrouter", "ADEP_MODEL_NAME": "meta-llama/llama-3.1-70b-instruct"}):
            config = AgentConfig()
            self.assertEqual(config.PROVIDER, "openrouter")
            self.assertEqual(config.MODEL_NAME, "meta-llama/llama-3.1-70b-instruct")
            self.assertTrue(isinstance(config.MODEL, LiteLlm))
            self.assertEqual(config.MODEL.model, "openrouter/meta-llama/llama-3.1-70b-instruct")

    def test_nvidia_config(self):
        with patch.dict(os.environ, {"ADEP_PROVIDER": "nvidia", "ADEP_MODEL_NAME": "meta/llama-3.1-nemotron-70b-instruct"}):
            config = AgentConfig()
            self.assertEqual(config.PROVIDER, "nvidia")
            self.assertEqual(config.MODEL_NAME, "meta/llama-3.1-nemotron-70b-instruct")
            self.assertTrue(isinstance(config.MODEL, LiteLlm))
            self.assertEqual(config.MODEL.model, "nvidia_nim/meta/llama-3.1-nemotron-70b-instruct")

if __name__ == '__main__':
    unittest.main()

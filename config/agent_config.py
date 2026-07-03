import os
from google.adk.models.lite_llm import LiteLlm

class AgentConfig:
    def __init__(self):
        # Read environment configurations
        # Provider options: 'google', 'ollama', 'openrouter', 'nvidia'
        self.PROVIDER = os.getenv("ADEP_PROVIDER", "google").lower()
        self.MODEL_NAME = os.getenv("ADEP_MODEL_NAME", "gemini-3.1-flash-lite")
        
    @property
    def MODEL(self):
        """Returns the configured model string or LiteLlm connector object."""
        if self.PROVIDER == "google":
            # Native Google Gemini integration
            return self.MODEL_NAME
            
        elif self.PROVIDER == "ollama":
            # Use 'ollama_chat' to prevent infinite tool loops
            return LiteLlm(model=f"ollama_chat/{self.MODEL_NAME}")
            
        elif self.PROVIDER == "openrouter":
            # OpenRouter model (e.g. meta-llama/llama-3.1-70b-instruct)
            return LiteLlm(model=f"openrouter/{self.MODEL_NAME}")
            
        elif self.PROVIDER == "nvidia":
            # NVIDIA Nemotron NIM endpoint
            return LiteLlm(model=f"nvidia/{self.MODEL_NAME}")
            
        else:
            # Fallback direct LiteLLM provider string
            return LiteLlm(model=self.MODEL_NAME)

config = AgentConfig()

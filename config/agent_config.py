import os
from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm

# Load environment variables from .env file
load_dotenv()

class AgentConfig:
    def __init__(self):
        # Read environment configurations
        # Provider options: 'google', 'ollama', 'openrouter', 'nvidia'
        self.PROVIDER = os.getenv("ADEP_PROVIDER", "nvidia").lower()
        self.MODEL_NAME = os.getenv("ADEP_MODEL_NAME", "nvidia/nemotron-3-nano-30b-a3b")

        
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
            return LiteLlm(model=f"nvidia_nim/{self.MODEL_NAME}")
            
        else:
            # Fallback direct LiteLLM provider string
            return LiteLlm(model=self.MODEL_NAME)

config = AgentConfig()

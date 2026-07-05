import os
import sys
import unittest
import asyncio
from unittest.mock import patch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.problem_identifier_agent import create_problem_identifier_agent
from utils.shared_environment import SHARED_GLOBALS
from google.adk.sessions import InMemorySessionService
from google.adk import Runner
from google.adk.models.lite_llm import LiteLlm

class TestProblemIdentification(unittest.TestCase):
    def setUp(self):
        SHARED_GLOBALS.clear()
        SHARED_GLOBALS['file_path'] = 'data/raw/raw_dataset.csv'

    def tearDown(self):
        SHARED_GLOBALS.clear()

    @patch('builtins.input')
    def test_supervised_auto_detect_price(self, mock_input):
        # Mock user selecting "price" and choice "1" (Auto-Detect)
        mock_input.side_effect = lambda prompt="": "price" if "target column" in prompt else "1"

        # Run within an async event loop
        async def run_agent():
            # Use OpenRouter/Gemini-2.5-flash with max_tokens=4000 to fit credit limits
            model = LiteLlm(model="openrouter/google/gemini-2.5-flash", max_tokens=4000)
            agent = create_problem_identifier_agent(model=model)
            
            session_service = InMemorySessionService()
            await session_service.create_session(app_name="ADEP", user_id="user_test", session_id="session_test")
            
            runner = Runner(agent=agent, app_name="ADEP", session_service=session_service)
            
            async for event in runner.run_async(user_id="user_test", session_id="session_test"):
                if event.is_final_response():
                    break

        asyncio.run(run_agent())

        # Verify that target_column and chosen_task were correctly processed and stored
        self.assertEqual(SHARED_GLOBALS.get('target_column'), 'price')
        self.assertEqual(SHARED_GLOBALS.get('chosen_task'), 'linear')

if __name__ == '__main__':
    unittest.main()

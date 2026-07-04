import os
import sys
import unittest
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.observability import TraceLogger

class TestTraceLogger(unittest.TestCase):
    def setUp(self):
        self.test_log_dir = "test_logs"
        self.logger = TraceLogger(log_dir=self.test_log_dir)

    def tearDown(self):
        # Remove temporary test logs folder after run
        if os.path.exists(self.test_log_dir):
            shutil.rmtree(self.test_log_dir)

    def test_log_creation(self):
        self.assertTrue(os.path.exists(self.test_log_dir))
        self.assertTrue(os.path.exists(self.logger.master_log_path))
        self.assertTrue(os.path.exists(self.logger.system_log_path))
        self.assertTrue(os.path.exists(self.logger.thinking_log_path))
        self.assertTrue(os.path.exists(self.logger.tool_log_path))

    def test_log_system(self):
        test_msg = "Starting data engineering run."
        self.logger.log_system("initialization", test_msg)
        
        with open(self.logger.system_log_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("[SYSTEM-INITIALIZATION] Starting data engineering run.", content)
            
        with open(self.logger.master_log_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("[SYSTEM-INITIALIZATION] Starting data engineering run.", content)

    def test_log_thinking(self):
        test_msg = "Evaluating target column cardinality."
        self.logger.log_thinking("ML_ARCHITECT", test_msg)
        
        with open(self.logger.thinking_log_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("[ML_ARCHITECT] Thinking: Evaluating target column cardinality.", content)

    def test_log_tool_call(self):
        self.logger.log_tool_call("CODING_AGENT", "execute_python_code", {"code": "print('hello')"})
        
        with open(self.logger.tool_log_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("print('hello')", content)

if __name__ == '__main__':
    unittest.main()

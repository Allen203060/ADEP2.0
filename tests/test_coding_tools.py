import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.coding_tools import execute_python_code, execute_python_subprocess
from utils.shared_environment import SHARED_GLOBALS

class TestCodingTools(unittest.TestCase):
    def test_execute_python_code_success(self):
        code = "print('Hello from test_execute_python_code_success')"
        result = execute_python_code(code)
        self.assertIn("Execution Success", result)
        self.assertIn("Hello from test_execute_python_code_success", result)

    def test_execute_python_code_memory_sharing(self):
        SHARED_GLOBALS["test_val"] = 42
        code = """
SHARED_GLOBALS['test_val'] = SHARED_GLOBALS['test_val'] + 10
print("Value is now:", SHARED_GLOBALS['test_val'])
"""
        result = execute_python_code(code)
        self.assertIn("Value is now: 52", result)
        self.assertEqual(SHARED_GLOBALS["test_val"], 52)
        
        # Clean up
        if "test_val" in SHARED_GLOBALS:
            SHARED_GLOBALS.clear()

    def test_execute_python_subprocess_success(self):
        code = "print('Hello Subprocess')"
        result = execute_python_subprocess(code)
        self.assertIn("Execution Success", result)
        self.assertIn("Hello Subprocess", result)

    def test_execute_python_code_syntax_error(self):
        code = "invalid python code string syntax"
        result = execute_python_code(code)
        self.assertIn("Execution Failed", result)

if __name__ == '__main__':
    unittest.main()

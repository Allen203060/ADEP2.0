import os
import sys
import unittest
import threading

# Add root folder to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.shared_environment import SharedEnvironment

class TestSharedEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = SharedEnvironment()

    def test_set_and_get(self):
        self.env.set("key1", "value1")
        self.assertEqual(self.env.get("key1"), "value1")
        self.assertEqual(self.env["key1"], "value1")

    def test_contains(self):
        self.env.set("key1", "value1")
        self.assertTrue("key1" in self.env)
        self.assertFalse("key2" in self.env)

    def test_clear(self):
        self.env.set("key1", "value1")
        self.env.clear()
        self.assertFalse("key1" in self.env)

    def test_thread_safety(self):
        # Run multiple threads setting keys concurrently to verify locking
        def worker(idx):
            for i in range(100):
                self.env.set(f"thread_{idx}_val_{i}", idx)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check all values written cleanly
        for idx in range(10):
            for i in range(100):
                self.assertEqual(self.env.get(f"thread_{idx}_val_{i}"), idx)

if __name__ == '__main__':
    unittest.main()

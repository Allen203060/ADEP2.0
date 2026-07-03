import os
import threading

class SharedEnvironment:
    """Thread-safe global state registry for cross-agent configuration."""
    def __init__(self):
        self._registry = {}
        self._lock = threading.Lock()

    def get(self, key, default=None):
        with self._lock:
            return self._registry.get(key, default)

    def set(self, key, value):
        with self._lock:
            self._registry[key] = value

    def clear(self):
        with self._lock:
            self._registry.clear()

    def __getitem__(self, key):
        with self._lock:
            return self._registry[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._registry[key] = value

    def __contains__(self, key):
        with self._lock:
            return key in self._registry

# Single instance importable by all files
SHARED_GLOBALS = SharedEnvironment()

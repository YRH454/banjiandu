"""CPU-only LRU invariants, including lookup-before-insert used by PrefixCache."""
import sys
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code"))
from perf_cache import TensorLRU


class FakeTensor:
    def __init__(self, count=1):
        self.count = count

    def numel(self):
        return self.count

    def element_size(self):
        return 4


class CacheTests(unittest.TestCase):
    def test_memory_bound(self):
        cache = TensorLRU(12)
        for i in range(100):
            cache[i] = FakeTensor()
        self.assertEqual(cache.used, 12)
        self.assertEqual(len(cache), 3)

    def test_hit_remains_after_missing_prefix_insert(self):
        cache = TensorLRU(12)
        for key in ("requested", "stale1", "stale2"):
            cache[key] = FakeTensor()
        self.assertIn("requested", cache)
        cache["missing"] = FakeTensor()
        self.assertIsInstance(cache["requested"], FakeTensor)
        self.assertNotIn("stale1", cache)

    def test_replacement_and_delete_accounting(self):
        cache = TensorLRU(20)
        cache["a"] = FakeTensor(3)
        cache["a"] = FakeTensor(2)
        self.assertEqual(cache.used, 8)
        del cache["a"]
        self.assertEqual(cache.used, 0)


if __name__ == "__main__":
    unittest.main()

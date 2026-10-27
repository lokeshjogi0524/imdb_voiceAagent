import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imdb_agent.llm import LLM


class ProviderError(Exception):
    def __init__(self, body):
        super().__init__("400 json_validate_failed")
        self.body = body


class LLMErrorHandlingTests(unittest.TestCase):
    def setUp(self):
        self.llm = LLM.__new__(LLM)
        self.llm.provider = "groq"

    def test_json_mode_refusal_becomes_refuse_action(self):
        error = ProviderError({"error": {
            "code": "json_validate_failed",
            "failed_generation": "I'm sorry, but I can only help with movie questions.",
        }})
        with patch.object(self.llm, "_generate", side_effect=error):
            result = json.loads(self.llm.generate("system", [], json_mode=True))

        self.assertEqual(result["action"], "refuse")
        self.assertEqual(result["args"], {})

    def test_other_json_validation_errors_still_raise(self):
        error = ProviderError({"error": {
            "code": "json_validate_failed",
            "failed_generation": "{not valid json",
        }})
        with patch.object(self.llm, "_generate", side_effect=error):
            with self.assertRaises(ProviderError):
                self.llm.generate("system", [], json_mode=True)

    def test_non_json_mode_refusal_still_raises(self):
        error = ProviderError({"error": {
            "code": "json_validate_failed",
            "failed_generation": "I'm sorry, but I cannot help with that.",
        }})
        with patch.object(self.llm, "_generate", side_effect=error):
            with self.assertRaises(ProviderError):
                self.llm.generate("system", [], json_mode=False)


if __name__ == "__main__":
    unittest.main()
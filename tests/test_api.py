import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import app.api as api_module
from app.services.token_tracker import TokenMetrics


class APITests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api_module.app)

    def test_health_returns_ok(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_chat_returns_typed_payload(self):
        fake_service = MagicMock()
        fake_service.answer_query.return_value = {
            "answer": "Article 1 defines the scope.",
            "sources": [
                {
                    "article": "Article 1",
                    "paragraph": "1",
                    "page": 2,
                    "text": "This Regulation lays down...",
                }
            ],
            "token_usage": {
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25,
            },
            "metrics": TokenMetrics(
                prompt_tokens=20,
                completion_tokens=5,
                total_tokens=25,
                bytes_in=1200,
                bytes_out=340,
            ),
            "judge": {
                "faithfulness_score": 0.9,
                "relevance_score": 0.95,
                "citation_integrity_score": 1.0,
                "hallucinations_detected": [],
                "citation_errors": [],
                "explanation": "Grounded answer.",
            },
        }

        with patch.object(api_module, "service", fake_service):
            response = self.client.post("/chat", json={"query": "What is Article 1?", "limit": 4})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["answer"], "Article 1 defines the scope.")
        self.assertIn("metrics", data)
        self.assertEqual(data["metrics"]["bytes_in"], 1200)
        self.assertEqual(data["judge"]["faithfulness_score"], 0.9)

    def test_chat_rejects_empty_query(self):
        response = self.client.post("/chat", json={"query": "   "})
        self.assertEqual(response.status_code, 400)

    def test_chat_returns_backend_error(self):
        fake_service = MagicMock()
        fake_service.answer_query.side_effect = RuntimeError("boom")

        with patch.object(api_module, "service", fake_service):
            response = self.client.post("/chat", json={"query": "hello"})

        self.assertEqual(response.status_code, 500)
        self.assertIn("Backend error", response.json().get("detail", ""))


if __name__ == "__main__":
    unittest.main()

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.services.rag_service import RAGService


class RAGFlowTests(unittest.TestCase):
    def test_answer_query_uses_retriever_and_returns_sources(self):
        service = RAGService.__new__(RAGService)
        service.client = MagicMock()
        service.model = "test-model"
        service.retriever = MagicMock()
        service.retriever.search.return_value = [
            {"article": "Article 1", "paragraph": "1", "page": 2, "text": "The act defines AI systems."}
        ]
        service.judge_service = MagicMock()
        service.judge_service.evaluate_response.return_value = {
            "faithfulness_score": 0.95,
            "relevance_score": 0.97,
            "explanation": "Grounded and relevant",
        }

        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Answer with citation [Source #1]."))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        )
        service.client.chat.completions.create.return_value = mock_response

        result = service.answer_query("What is Article 1?")

        self.assertIn("Answer with citation", result["answer"])
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["article"], "Article 1")
        self.assertIn("judge", result)
        self.assertEqual(result["judge"]["faithfulness_score"], 0.95)


if __name__ == "__main__":
    unittest.main()

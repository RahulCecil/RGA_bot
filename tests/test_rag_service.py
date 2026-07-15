import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.services.rag_service import RAGService


class RAGServiceTests(unittest.TestCase):
    def test_generate_answer_with_citations_returns_metrics(self):
        service = RAGService.__new__(RAGService)
        service.client = MagicMock()
        service.model = "test-model"

        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Answer with citation [Source #1]."))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        )
        service.client.chat.completions.create.return_value = mock_response

        result = service.generate_answer_with_citations(
            "What does Article 1 say?",
            [{"article": "Article 1", "paragraph": "1", "page": 2, "text": "The act defines AI systems."}],
        )

        self.assertIn("Answer with citation", result["answer"])
        self.assertEqual(result["metrics"].prompt_tokens, 10)
        self.assertEqual(result["metrics"].completion_tokens, 2)


if __name__ == "__main__":
    unittest.main()

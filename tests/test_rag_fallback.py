import unittest
from unittest.mock import MagicMock

from app.services.rag_service import RAGService


class RAGFallbackTests(unittest.TestCase):
    def test_generate_answer_with_citations_falls_back_when_model_errors(self):
        service = RAGService.__new__(RAGService)
        service.client = MagicMock()
        service.model = "test-model"
        service.retriever = None
        service.judge_service = None

        service.client.chat.completions.create.side_effect = Exception("model unavailable")

        result = service.generate_answer_with_citations(
            "What does the EU AI Act say?",
            [{"article": "Article 1", "paragraph": "1", "page": 10, "text": "High-risk AI systems must be safe."}],
        )

        self.assertIn("local fallback", result["answer"].lower())
        self.assertEqual(result["metrics"].prompt_tokens, 0)


if __name__ == "__main__":
    unittest.main()

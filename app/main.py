import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.services.rag_service import RAGService


def main() -> None:
    print("RGA Bot is ready. Enter your question about the EU AI Act (or type 'exit').")
    service = RAGService()

    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not query:
            continue

        result = service.answer_query(query)
        print(f"\nAssistant:\n{result['answer']}\n")

        judge = result.get("judge", {})
        if judge:
            print("Judge score:")
            print(
                f"- Faithfulness: {judge.get('faithfulness_score', 'n/a')}"
            )
            print(f"- Relevance: {judge.get('relevance_score', 'n/a')}")
            if judge.get("explanation"):
                print(f"- Explanation: {judge.get('explanation')}")

        print("Sources:")
        for source in result.get("sources", []):
            article = source.get("article", "Unknown")
            paragraph = source.get("paragraph") or "n/a"
            page = source.get("page", "?")
            print(f"- {article} | paragraph {paragraph} | page {page}")


if __name__ == "__main__":
    main()

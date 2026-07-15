class JudgeService:
    def __init__(self, rag_service: RAGService):
        self.rag = rag_service

    def evaluate_response(self, query: str, context: str, response: str) -> dict:
        judge_prompt = f"""
        Analyze the following RAG system output based on the provided context.
        
        [Query]: {query}
        [Retrieved Context]: {context}
        [System Response]: {response}
        
        Evaluate the response based on:
        1. Faithfulness (Is the answer entirely supported by the context without hallucination?)
        2. Relevance (Does it answer the exact user query?)
        
        Respond ONLY with a JSON payload in this structure:
        {{
            "faithfulness_score": <float 0.0 to 1.0>,
            "relevance_score": <float 0.0 to 1.0>,
            "explanation": "<brief reason for the score>"
        }}
        """
        # Querying the judge model
        res = self.rag.client.chat.completions.create(
            model=self.rag.model,
            messages=[{"role": "user", "content": judge_prompt}],
            response_format={"type": "json_object"}
        )
        return res.choices[0].message.content
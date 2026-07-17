import json
import re


class JudgeService:
    def __init__(self, rag_service):
        self.rag = rag_service

    def evaluate_response(self, query: str, context_chunks: list, response: str) -> dict:
        """
        Evaluates the generated chatbot response against the retrieved database chunks.
        
        :param query: The original user search query.
        :param context_chunks: A list of dicts from retrieval containing keys like:
                       [{'text'|'content': ..., 'article': ..., 'paragraph'|'paragraph_number': ..., 'page'|'page_number': ...}]
        :param response: The generated chatbot response text containing markdown citations e.g. [Source #1]
        """
        # 1. Parse and compile the actual citations that WERE sent to the generator
        valid_citations_manifest = []
        formatted_sources = []
        
        for i, chunk in enumerate(context_chunks):
            art = chunk.get("article", "Unknown Article")
            para = chunk.get("paragraph_number", chunk.get("paragraph"))
            pg = chunk.get("page_number", chunk.get("page", "Unknown Page"))
            
            para_label = f"Paragraph {para}" if para is not None else "General Text"
            citation_string = f"[{art}, {para_label}, Page {pg}]"
            
            valid_citations_manifest.append({
                "article": str(art),
                "paragraph_number": para,
                "page_number": pg,
                "citation_format": citation_string
            })
            
            # Format cleanly for the LLM's raw prompt
            formatted_sources.append(
                f"Source ID {i+1}:\n"
                f"Citation Metadata: {citation_string}\n"
                f"Source Content:\n{chunk.get('content', chunk.get('text', ''))}\n"
                f"----------------"
            )

        context_text = "\n\n".join(formatted_sources)

        # 2. Structure the deep compliance auditing prompt
        judge_prompt = f"""
        You are an elite, highly pedantic Legal Audit Judge evaluating an AI system's answer on the EU AI Act.
        Your job is to verify that the system response matches the actual legal sources without stretching, misinterpreting, or hallucinating references.
        
        [Query]: {query}
        
        [Retrieved Context Blocks]:
        {context_text}
        
        [Generated System Response to Audit]: 
        {response}
        
        EVALUATION CRITERIA:
        1. Faithfulness (Score 0.0 to 1.0): Are the factual claims completely true to the provided context? Deduct heavily if the response introduces outside information, exaggerates, or switches modal verbs (e.g., changing 'must' to 'may').
        2. Relevance (Score 0.0 to 1.0): Does it directly address the query?
        3. Citation Integrity (Score 0.0 to 1.0): Did the response correctly match its factual claims to the corresponding citation metadata (Article, Paragraph, Page) provided in the sources?
        
        Respond ONLY with a JSON payload in this structure (do not add any markdown styling outside the JSON):
        {{
            "faithfulness_score": <float 0.0 to 1.0>,
            "relevance_score": <float 0.0 to 1.0>,
            "citation_integrity_score": <float 0.0 to 1.0>,
            "hallucinations_detected": [
                "List specific claims in the response that are NOT supported by the text"
            ],
            "citation_errors": [
                "List any instances where a claim is cited using a Page, Article, or Paragraph number that does not match the source content"
            ],
            "explanation": "<overall logical reasoning for scores>"
        }}
        """

        # 3. Native Post-Processing: Extract citations via Python Regex for a hard validation layer.
        # The answer generator requires inline citations in the format [Source #X].
        response_citations = re.findall(r'\[Source\s*#\s*(\d+)\]', response, re.IGNORECASE)
        
        hard_citation_failures = []
        max_source_index = len(valid_citations_manifest)
        for source_num_text in response_citations:
            try:
                source_num = int(source_num_text)
            except ValueError:
                hard_citation_failures.append(
                    f"Malformed source citation found in output: [Source #{source_num_text}]"
                )
                continue

            if source_num < 1 or source_num > max_source_index:
                hard_citation_failures.append(
                    f"Hallucinated source citation found in output: [Source #{source_num}] "
                    f"is outside retrieved source range 1..{max_source_index}."
                )

        try:
            res = self.rag.client.chat.completions.create(
                model=self.rag.model,
                messages=[{"role": "user", "content": judge_prompt}],
                response_format={"type": "json_object"}
            )
            content = res.choices[0].message.content or "{}"
            result_payload = json.loads(content)
            
            # Inject our deterministic Python hard check into the LLM payload
            if hard_citation_failures:
                result_payload["citation_errors"] = list(
                    set(result_payload.get("citation_errors", []) + hard_citation_failures)
                )
                # Tank the citation score to 0.0 if there is a flat-out hallucinated coordinate
                result_payload["citation_integrity_score"] = 0.0
                
            return result_payload

        except Exception as e:
            return {
                "faithfulness_score": 0.0,
                "relevance_score": 0.0,
                "citation_integrity_score": 0.0,
                "hallucinations_detected": [],
                "citation_errors": [f"Evaluation crash: {str(e)}"],
                "explanation": "Judge execution failed or returned unparseable JSON.",
            }
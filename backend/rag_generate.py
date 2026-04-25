#!/usr/bin/env python3
"""
RAG Generation Pipeline for SIGGRAPH 2025 Papers.

Uses the retrieval pipeline to find relevant chunks,
then generates an answer using an LLM via OpenRouter API.

Usage:
    from rag_generate import RAGGenerator, GenerationConfig, SYSTEM_PROMPT
    
    generator = RAGGenerator()
    result = generator.generate("What is 3D Gaussian Splatting?")
    print(result["answer"])
"""

import os
import requests
from typing import Optional
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

from retrieval_pipeline import RetrievalPipeline, RetrievalResult


# =============================================================================
# SYSTEM PROMPT - This tells the LLM how to behave
# =============================================================================
SYSTEM_PROMPT = """You are an expert research assistant specializing in computer graphics, specifically SIGGRAPH 2025 papers.

Your task is to answer questions using ONLY the provided research paper excerpts.

Rules:
1. Cite sources using [Paper Title] format
2. Be comprehensive and technically accurate
3. If the excerpts don't contain the answer, say so
4. Use LaTeX for math: $inline$ or $$block$$
5. Do NOT make up information not in the excerpts
6. Do NOT include a References section at the end
"""


# =============================================================================
# QUERY REFINEMENT PROMPT
# =============================================================================
QUERY_REFINEMENT_PROMPT = """You are an expert at refining search queries for academic paper retrieval.

Given a user's question, rewrite it as a clear, focused search query that will retrieve the most relevant research papers.

Keep it concise (under 20 words). Focus on key technical terms.

User question: {query}

Refined search query:"""


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class GenerationConfig:
    """Configuration for the RAG generator."""
    llm_model: str = "openai/gpt-4o"          # Model for answer generation
    llm_provider: str = "openrouter"     
    temperature: float = 0.1                    # Low temperature = more factual
    max_tokens: int = 2000                      # Max length of generated answer
    openrouter_api_key: Optional[str] = None    # Will load from env if not set
    refine_query: bool = True                   # Whether to refine queries
    refinement_model: str = "openai/gpt-3.5-turbo"  # Cheaper model for refinement
    retrieval_top_k: int = 8                    # Number of chunks to retrieve
    use_reranker: bool = True


# =============================================================================
# RAG GENERATOR CLASS
# =============================================================================
class RAGGenerator:
    """
    Main RAG class - this is what api_server.py uses!

    Flow:
    1. Refine the user's query (optional — makes retrieval better)
    2. Retrieve relevant chunks using the retrieval pipeline
    3. Format chunks into a context string
    4. Generate answer using LLM
    5. Return answer with source metadata for citations
    """

    def __init__(self, config: Optional[GenerationConfig] = None, retrieval_pipeline=None):
        """
        Initialize the RAG generator.

        Sets up config, retrieval pipeline, and API credentials.
        """
        # use default config if none provided
        self.config = config or GenerationConfig()

        # use provided pipeline or create a new one from environment variables
        # RetrievalPipeline connects to Qdrant and builds the BM25 index
        self.retrieval = retrieval_pipeline or RetrievalPipeline()

        # get OpenRouter API key from config or environment
        self.openrouter_api_key = (
            self.config.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        )

        # validate API key exists — we can't call the LLM without it
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not set in .env or config")

        # base URL for all OpenRouter API calls
        self.openrouter_base_url = "https://openrouter.ai/api/v1"

        print("RAG Generator initialized successfully!")

    def refine_query(self, query: str) -> str:
        """
        Use LLM to improve the search query before retrieval.

        Example:
        original: "how does gaussian splatting work?"
        refined:  "3D Gaussian Splatting rendering technique neural radiance"

        The refined query uses better technical terms that match paper content.
        """
        # if refinement is disabled, just return the original query
        if not self.config.refine_query:
            return query

        try:
            # build the refinement prompt
            prompt = QUERY_REFINEMENT_PROMPT.format(query=query)

            # headers for OpenRouter API authentication
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json"
            }

            # payload — use cheaper model for refinement to save costs
            payload = {
                "model": self.config.refinement_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,   # slightly creative for query rewriting
                "max_tokens": 100     # refined query should be short
            }

            # call OpenRouter API
            response = requests.post(
                f"{self.openrouter_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=10  # don't wait too long for refinement
            )

            # if request failed, fall back to original query
            if response.status_code != 200:
                print(f"Query refinement failed (status {response.status_code}), using original query")
                return query

            # extract the refined query from response
            response_json = response.json()
            refined = response_json["choices"][0]["message"]["content"].strip()

            # remove any quotes the model might have added
            refined = refined.strip('"\'')

            print(f"Query refined: '{query}' → '{refined}'")
            return refined

        except Exception as e:
            # if anything goes wrong, use original query — don't crash!
            print(f"Query refinement error: {e}, using original query")
            return query

    def _format_context(self, results: list[RetrievalResult]) -> str:
        """
        Format retrieved chunks into a context string for the LLM.

        Takes the retrieved paper chunks and formats them so the LLM
        can read them and use them to answer the question.

        Example output:
        --- Source 1 ---
        Title: 3D Gaussian Splatting for Real-Time...
        Authors: Kerbl et al.
        Content: 3D Gaussian Splatting represents scenes as...
        """
        sources = []

        for i, result in enumerate(results, 1):
            # format each chunk as a clearly labelled source
            formatted = f"""
--- Source {i} ---
Title: {result.title}
Authors: {result.authors}
Section: {result.chunk_section}

Content:
{result.text}
"""
            sources.append(formatted)

        # join all sources into one big context string
        return "\n".join(sources)

    def _build_sources_metadata(self, results: list[RetrievalResult]) -> list[dict]:
        """
        Build list of unique source papers for citations.

        The frontend displays these as clickable links to PDFs, GitHub repos,
        and videos. We deduplicate by title so the same paper doesn't appear twice.
        """
        # track seen titles to avoid duplicate papers in citations
        seen = {}

        for result in results:
            # only add each paper once (multiple chunks can come from same paper)
            if result.title not in seen:
                seen[result.title] = {
                    "title": result.title,
                    "authors": result.authors,
                    "pdf_url": result.pdf_url,
                    "github_link": result.github_link,
                    "video_link": result.video_link,
                    "acm_url": result.acm_url,
                    "abstract_url": result.abstract_url,
                }

        return list(seen.values())

    def _call_llm(self, query: str, context: str) -> str:
        """
        Call OpenRouter API to generate an answer using the retrieved context.

        Sends the system prompt + context + question to the LLM.
        The LLM reads the paper excerpts and generates a cited answer.
        """
        # build the user message with context and question
        user_message = f"""Based on the following research paper excerpts, answer this question.

Question: {query}

Research Paper Excerpts:
{context}

Remember to cite papers using [Paper Title] format."""

        # headers for OpenRouter API
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }

        # payload — system prompt sets behaviour, user message has the question
        payload = {
            "model": self.config.llm_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            "temperature": self.config.temperature,  # low = more factual
            "max_tokens": self.config.max_tokens
        }

        print(f"Calling LLM ({self.config.llm_model})...")

        # call OpenRouter API
        response = requests.post(
            f"{self.openrouter_base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60  # LLM can take a while for long answers
        )

        # check if the API call succeeded
        if response.status_code != 200:
            raise ValueError(
                f"LLM API error {response.status_code}: {response.text}"
            )

        # extract the generated answer from the response
        response_json = response.json()
        answer = response_json["choices"][0]["message"]["content"]

        return answer

    def generate(self, query: str, top_k: Optional[int] = None, return_sources: bool = True) -> dict:
        """
        Full RAG pipeline - THIS IS THE MAIN METHOD api_server.py CALLS!

        Flow:
        query → refine → retrieve chunks → format context → LLM → answer + citations
        """
        print(f"\nGenerating answer for: '{query}'")

        # step 1: refine the query to improve retrieval quality
        refined = self.refine_query(query)

        # step 2: retrieve relevant chunks from Qdrant + BM25
        k = top_k or self.config.retrieval_top_k
        print(f"Retrieving top {k} chunks...")
        results = self.retrieval.retrieve(refined, top_k=k)

        # step 3: handle case where no results found
        if not results:
            return {
                "query": query,
                "refined_query": refined,
                "answer": "I couldn't find any relevant papers to answer this question.",
                "sources": []
            }

        print(f"Retrieved {len(results)} chunks from {len(set(r.title for r in results))} papers")

        # step 4: format retrieved chunks into context string for LLM
        context = self._format_context(results)

        # step 5: generate answer using LLM with context
        answer = self._call_llm(refined, context)

        print("Answer generated successfully!")

        # step 6: return complete response with answer and source citations
        return {
            "query": query,
            "refined_query": refined,
            "answer": answer,
            "sources": self._build_sources_metadata(results) if return_sources else []
        }


# =============================================================================
# CLI FOR TESTING
# =============================================================================
if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "What is 3D Gaussian Splatting?"

    print("Initializing RAG Generator...")
    generator = RAGGenerator()

    print(f"\nQuery: {query}")
    print("=" * 60)

    result = generator.generate(query)

    print(f"Refined Query: {result.get('refined_query', 'N/A')}")
    print("=" * 60)
    print("\nAnswer:")
    print(result['answer'])
    print("=" * 60)
    print(f"\nSources: {len(result.get('sources', []))} papers")
    for source in result.get('sources', []):
        print(f"  - {source['title']}")
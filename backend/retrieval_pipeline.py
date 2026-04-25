#!/usr/bin/env python3
"""
Retrieval Pipeline for SIGGRAPH 2025 Papers.

Implements hybrid search:
1. Semantic search (embeddings via OpenRouter + Qdrant Cloud)
2. Keyword search (BM25 - runs locally)
3. Reranking (Cohere API - optional)

Usage:
    from retrieval_pipeline import RetrievalPipeline
    
    pipeline = RetrievalPipeline()
    results = pipeline.retrieve("3D Gaussian Splatting", top_k=5)
"""

import json
import os
import re
import requests
import numpy as np
from typing import Optional, List
from dataclasses import dataclass
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from dotenv import load_dotenv
load_dotenv()

# Must match the collection name used in upload_to_qdrant.py
COLLECTION_NAME = "siggraph2025_papers"


@dataclass
class RetrievalResult:
    """
    Represents a single search result.
    The api_server.py expects these exact fields - do not change!
    """
    chunk_id: str
    paper_id: str
    title: str
    authors: str
    text: str
    score: float
    chunk_type: str = ""
    chunk_section: str = ""
    pdf_url: Optional[str] = None
    github_link: Optional[str] = None
    video_link: Optional[str] = None
    acm_url: Optional[str] = None
    abstract_url: Optional[str] = None


@dataclass
class RetrievalPipelineConfig:
    """Configuration for the retrieval pipeline."""
    qdrant_url: str
    qdrant_api_key: str
    openrouter_api_key: str
    embedding_model: str = "baai/bge-large-en-v1.5"
    chunks_path: str = "./chunks.json"
    semantic_weight: float = 0.7
    bm25_weight: float = 0.3
    use_reranker: bool = True
    cohere_api_key: Optional[str] = None


class OpenRouterEmbedder:
    """
    Generate embeddings using OpenRouter API.
    Used to embed user queries for semantic search.
    """
    
    def __init__(self, api_key: str, model: str = "baai/bge-large-en-v1.5"):
        """
        Initialize the embedder.
        Stores API key, model name, and base URL.
        """
        # store the api key so we can use it in embed_query
        self.api_key = api_key
        # store the model name — must match what was used to create embeddings!
        self.model = model
        # base URL for OpenRouter API
        self.base_url = "https://openrouter.ai/api/v1"
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Takes text → sends to OpenRouter API → gets back a 1024-dimensional vector
        This vector represents the MEANING of the text, not just the words.
        """
        # build headers — Authorization tells OpenRouter who we are
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # build payload — what model to use and what text to embed
        payload = {
            "model": self.model,
            "input": text
        }
        
        # make the API call to OpenRouter's embeddings endpoint
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json=payload
        )
        
        # check if the API call succeeded
        if response.status_code != 200:
            raise ValueError(f"OpenRouter API error {response.status_code}: {response.text}")
        
        # parse the response JSON
        response_data = response.json()
        
        # extract the embedding vector from the response
        # response_data["data"][0]["embedding"] is a list of 1024 floats
        embedding = response_data["data"][0]["embedding"]
        
        # convert to numpy array (float32 is more memory efficient)
        return np.array(embedding, dtype=np.float32)


class BM25Index:
    """
    BM25 index for keyword search.
    This runs entirely locally - no API calls needed!
    
    BM25 is a classic keyword search algorithm — it finds documents
    that contain the exact words in your query. Great for specific
    technical terms that semantic search might miss!
    """
    
    def __init__(self, chunks: list[dict]):
        """
        Build BM25 index from chunks.
        
        Think of this like building a search index —
        we process all 11,008 chunks once upfront so searches are fast.
        """
        # store chunks so we can look them up later
        self.chunks = chunks
        
        # create a lookup dict: chunk_id → index in chunks list
        # this lets us quickly find a chunk given its ID
        self.chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}
        
        # tokenize all documents — convert each chunk's text to a list of words
        # BM25 works on word tokens, not raw text
        print(f"Tokenizing {len(chunks)} chunks for BM25 index...")
        self.tokenized_docs = [self._tokenize(c["text"]) for c in chunks]
        
        # build the BM25 index from the tokenized documents
        # BM25Okapi is a popular variant of BM25
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print("BM25 index built successfully!")
    
    def _tokenize(self, text: str) -> list[str]:
        """
        Simple tokenization: lowercase and extract alphanumeric words.
        
        Example:
        "3D Gaussian Splatting" → ["3d", "gaussian", "splatting"]
        """
        # convert to lowercase — so "Gaussian" matches "gaussian"
        text = text.lower()
        
        # find all alphanumeric sequences (words and numbers)
        # re.findall returns a list of all matches
        tokens = re.findall(r'[a-z0-9]+', text)
        
        return tokens
    
    def search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        """
        Search for query and return top-k results.
        
        Returns list of (chunk_index, score) tuples sorted by score descending.
        Only returns results with non-zero scores (i.e. actual matches).
        """
        # tokenize the query the same way we tokenized the documents
        query_tokens = self._tokenize(query)
        
        # get BM25 scores for ALL documents
        # scores[i] = how relevant document i is to the query
        scores = self.bm25.get_scores(query_tokens)
        
        # get indices of top-k highest scores
        # argsort sorts ascending, so we take the last top_k elements reversed
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # build result list — only include results with non-zero scores
        results = [
            (int(idx), float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]
        
        return results


class RetrievalPipeline:
    """
    Main retrieval pipeline combining semantic search + BM25 + reranking.
    This is what api_server.py uses to find relevant chunks.
    
    Flow:
    query → semantic search (find semantically similar chunks)
          + BM25 search (find keyword matching chunks)
          → combine scores (hybrid search)
          → rerank (optional, improves quality)
          → return top results
    """
    
    def __init__(self, config: Optional[RetrievalPipelineConfig] = None):
        """
        Initialize all components of the retrieval pipeline.
        """
        # if no config provided, load from environment variables
        if config is None:
            config = RetrievalPipelineConfig(
                qdrant_url=os.getenv("QDRANT_URL"),
                qdrant_api_key=os.getenv("QDRANT_API_KEY"),
                openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
                cohere_api_key=os.getenv("COHERE_API_KEY"),
                chunks_path=os.getenv("CHUNKS_PATH", "./chunks.json"),
            )
        
        # validate required fields — we can't work without these
        if not config.qdrant_url:
            raise ValueError("QDRANT_URL not set in .env")
        if not config.qdrant_api_key:
            raise ValueError("QDRANT_API_KEY not set in .env")
        if not config.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not set in .env")
        
        # initialize Qdrant client — connects to our vector database
        print(f"Connecting to Qdrant at {config.qdrant_url}...")
        self.qdrant = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key
        )
        print("Connected to Qdrant!")
        
        # initialize the embedder — used to embed user queries
        self.embedder = OpenRouterEmbedder(
            api_key=config.openrouter_api_key,
            model=config.embedding_model
        )
        print(f"Embedder initialized with model: {config.embedding_model}")
        
        # load chunks from JSON file
        print(f"Loading chunks from {config.chunks_path}...")
        with open(config.chunks_path, "r") as f:
            data = json.load(f)
        # handle both formats: direct list or dict with "chunks" key    
        if isinstance(data, list):
            self.chunks = data
        else:
            self.chunks = data["chunks"]
        print(f"Loaded {len(self.chunks)} chunks!")

        
        # build BM25 index from chunks
        self.bm25_index = BM25Index(self.chunks)
        
        # store the config for later use
        self.config = config
        
        print("Retrieval pipeline ready!")
    
    def semantic_search(self, query: str, top_k: int = 30) -> list[dict]:
        """
        Perform semantic search using Qdrant.
        
        Converts query to embedding → finds most similar vectors in Qdrant.
        This finds chunks that are SEMANTICALLY similar even if they don't
        use the exact same words.
        """
        # embed the query — converts text to a 1024-dimensional vector
        query_embedding = self.embedder.embed_query(query)
        
        # search Qdrant for the most similar vectors
        results = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding.tolist(),
            limit=top_k,
            with_payload=True  # include the chunk metadata in results
        ).points
        
        # convert Qdrant results to our standard dict format
        return [
            {
                "chunk_id": r.payload["chunk_id"],
                "score": r.score,
                "payload": r.payload
            }
            for r in results
        ]
    
    def bm25_search(self, query: str, top_k: int = 30) -> list[dict]:
        """
        Perform BM25 keyword search.
        
        Finds chunks that contain the exact keywords in the query.
        Great for specific technical terms like "3D Gaussian Splatting"
        that semantic search might miss.
        """
        # run BM25 search — returns (chunk_index, score) tuples
        results = self.bm25_index.search(query, top_k)
        
        # convert to our standard dict format (same as semantic_search)
        return [
            {
                "chunk_id": self.chunks[idx]["chunk_id"],
                "score": score,
                "payload": self.chunks[idx]
            }
            for idx, score in results
        ]
    
    def hybrid_search(self, query: str, semantic_top_k: int = 30, bm25_top_k: int = 30) -> list[dict]:
        """
        Combine semantic and BM25 results using weighted scoring.
        
        Why combine both?
        - Semantic search finds conceptually related chunks
        - BM25 finds exact keyword matches
        - Together they give better results than either alone!
        
        Formula:
        combined_score = 0.7 × semantic_score + 0.3 × bm25_score
        """
        # get results from both search methods
        semantic_results = self.semantic_search(query, semantic_top_k)
        bm25_results = self.bm25_search(query, bm25_top_k)
        
        # normalize semantic scores so they're between 0 and 1
        if semantic_results:
            max_semantic = max(r["score"] for r in semantic_results)
            for r in semantic_results:
                r["normalized_score"] = r["score"] / max_semantic if max_semantic > 0 else 0
        
        # normalize BM25 scores the same way
        if bm25_results:
            max_bm25 = max(r["score"] for r in bm25_results)
            for r in bm25_results:
                r["normalized_score"] = r["score"] / max_bm25 if max_bm25 > 0 else 0
        
        # combine results into a single dict keyed by chunk_id
        combined = {}
        
        # add semantic results
        for r in semantic_results:
            chunk_id = r["chunk_id"]
            combined[chunk_id] = {
                "chunk_id": chunk_id,
                "payload": r["payload"],
                "semantic_score": r["normalized_score"],
                "bm25_score": 0.0,
                # weighted combination: semantic is 70% of score
                "combined_score": self.config.semantic_weight * r["normalized_score"]
            }
        
        # add BM25 results — if chunk already found by semantic, add BM25 score
        for r in bm25_results:
            chunk_id = r["chunk_id"]
            if chunk_id in combined:
                # chunk found by BOTH methods — add BM25 contribution
                combined[chunk_id]["bm25_score"] = r["normalized_score"]
                combined[chunk_id]["combined_score"] += self.config.bm25_weight * r["normalized_score"]
            else:
                # chunk found ONLY by BM25
                combined[chunk_id] = {
                    "chunk_id": chunk_id,
                    "payload": r["payload"],
                    "semantic_score": 0.0,
                    "bm25_score": r["normalized_score"],
                    # weighted combination: BM25 is 30% of score
                    "combined_score": self.config.bm25_weight * r["normalized_score"]
                }
        
        # sort by combined_score descending (best results first)
        results = sorted(combined.values(), key=lambda x: x["combined_score"], reverse=True)
        
        return results
    
    def rerank(self, query: str, results: list[dict], top_k: int = 10) -> list[dict]:
        """
        Rerank results using Cohere API (optional but improves quality).
        
        Cohere's reranker is a specialized model that reads the query AND
        each document together to score relevance — much more accurate than
        just vector similarity!
        """
        # if no Cohere key or no results, just return top_k results as-is
        if not self.config.cohere_api_key or not results:
            return results[:top_k]
        
        try:
            # extract text from each result for Cohere to read
            texts = [r["payload"]["text"] for r in results]
            
            # call Cohere Rerank API
            response = requests.post(
                "https://api.cohere.ai/v1/rerank",
                headers={
                    "Authorization": f"Bearer {self.config.cohere_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "rerank-english-v3.0",
                    "query": query,
                    "documents": texts,
                    "top_n": top_k
                }
            )
            
            if response.status_code != 200:
                print(f"Cohere rerank failed: {response.text}, falling back to original order")
                return results[:top_k]
            
            # parse Cohere's response
            rerank_data = response.json()
            
            # reorder results based on Cohere's ranking
            reranked = []
            for item in rerank_data["results"]:
                result = results[item["index"]].copy()
                # add Cohere's relevance score
                result["rerank_score"] = item["relevance_score"]
                reranked.append(result)
            
            return reranked
            
        except Exception as e:
            # if anything goes wrong, fall back to original order
            print(f"Cohere reranking error: {e}, falling back to original order")
            return results[:top_k]
    
    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievalResult]:
        """
        Full retrieval pipeline - THIS IS WHAT api_server.py CALLS!
        
        Flow:
        query → hybrid search → rerank → top k results → RetrievalResult objects
        """
        print(f"Retrieving results for: '{query}'")
        
        # step 1: run hybrid search to get candidates
        candidates = self.hybrid_search(query)
        print(f"Hybrid search returned {len(candidates)} candidates")
        
        # step 2: rerank if enabled
        if self.config.use_reranker:
            reranked = self.rerank(query, candidates, top_k=min(top_k * 2, len(candidates)))
        else:
            reranked = candidates
        
        # step 3: take top_k results
        final = reranked[:top_k]
        print(f"Returning top {len(final)} results")
        
        # step 4: convert to RetrievalResult objects
        # api_server.py expects these exact objects!
        return [
            RetrievalResult(
                chunk_id=r["payload"]["chunk_id"],
                paper_id=r["payload"]["paper_id"],
                title=r["payload"]["title"],
                authors=r["payload"]["authors"],
                text=r["payload"]["text"],
                score=r.get("rerank_score", r.get("combined_score", r.get("score", 0))),
                chunk_type=r["payload"].get("chunk_type", ""),
                chunk_section=r["payload"].get("chunk_section", ""),
                pdf_url=r["payload"].get("pdf_url"),
                github_link=r["payload"].get("github_link"),
                video_link=r["payload"].get("video_link"),
                acm_url=r["payload"].get("acm_url"),
                abstract_url=r["payload"].get("abstract_url"),
            )
            for r in final
        ]


# For testing this file directly
if __name__ == "__main__":
    import sys
    
    query = sys.argv[1] if len(sys.argv) > 1 else "3D Gaussian Splatting"
    
    print(f"Testing retrieval pipeline with query: '{query}'")
    print("=" * 60)
    
    pipeline = RetrievalPipeline()
    results = pipeline.retrieve(query, top_k=5)
    
    print(f"\nFound {len(results)} results:\n")
    
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.score:.4f}] {r.title[:60]}...")
        print(f"   Paper ID: {r.paper_id}")
        print(f"   Text preview: {r.text[:100]}...")
        print()
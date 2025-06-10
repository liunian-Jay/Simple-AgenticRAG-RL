import json
import json
import os
import warnings
from typing import List, Dict, Optional
import argparse
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
from pyserini.search.lucene import LuceneSearcher

def load_indexR(index_path):
    lucene_searcher = LuceneSearcher(index_path)
    return lucene_searcher


def bm25_search(query, searcher, num_retrieval):
    results = []
    hits = searcher.search(query, num_retrieval)
    for item in hits:
        doc = searcher.doc(item.docid)
        json_doc = json.loads(doc.raw())

        contents = json_doc['contents']
        parts = contents.split('\n', 1)
        title = parts[0].strip('\"')
        text = parts[1]
        result = {
            'id': json_doc['id'],
            'title': title,
            'text': text,
            'score': float(item.score),
        }
        results.append(result)
    return results




class QueryRequest(BaseModel):
    query: str
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "query": "What is Python?"
      "topk": 3,
      "return_scores": true
    }
    """
    if not request.topk:
        request.topk = args.retrieval_topk  # fallback to default

    # Perform batch retrieval
    results = bm25_search(request.query, searcher, request.topk)
    return {"result": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the local retriever.")
    parser.add_argument("--index_path", type=str, default="/home/yjiang/myWork/retrieval/corpus/bm25/lucene-index.wikipedia-dpr-100w.20210120.d1b9e6", help="Corpus indexing file.")
    parser.add_argument("--corpus_path", type=str, default="/", help="Local corpus file.")
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    args = parser.parse_args()

    # 2) Instantiate a global retriever so it is loaded once and reused.
    searcher = load_indexR(args.index_path)
    
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

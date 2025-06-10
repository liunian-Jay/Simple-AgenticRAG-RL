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
from pyserini.search.faiss import FaissSearcher
from pyserini.encode import DprQueryEncoder

def load_corpus(corpus_path):
    lucene_searcher = LuceneSearcher(corpus_path)
    return lucene_searcher

def load_model(model_path):
    encoder = DprQueryEncoder(model_path)
    return encoder

def load_searcher(model, index_path):
    searcher = FaissSearcher(
        index_path,
        model,  
    )
    return searcher

def dpr_search(query, searcher, corpus, num_retrieval):
    results = []
    hits = searcher.search(query, num_retrieval)
    for item in hits:
        doc = corpus.doc(item.docid)
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
    Endpoint that accepts query and performs retrieval.
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
    results = dpr_search(request.query, searcher, request.topk)
    return {"result": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the local retriever.")
    parser.add_argument("--model_path", type=str, default="/home/yjiang/myWork/retrieval/models/dpr-question_encoder-multiset-base", help="Model path.")
    parser.add_argument("--index_path", type=str, default="/home/yjiang/myWork/retrieval/corpus/dpr_multi/faiss.wikipedia-dpr-100w.dpr_multi.20200127.f403c3", help="Corpus indexing file.")
    parser.add_argument("--corpus_path", type=str, default="/home/yjiang/myWork/retrieval/corpus/bm25/lucene-index.wikipedia-dpr-100w.20210120.d1b9e6", help="Local corpus file.")
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    args = parser.parse_args()

    # 2) Instantiate a global retriever so it is loaded once and reused.
    model = load_model(args.model_path)
    corpus = load_corpus(args.corpus_path)
    searcher = load_searcher(model, args.index_path)
    
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

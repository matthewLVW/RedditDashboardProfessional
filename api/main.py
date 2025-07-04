# api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nlp_core.embeddings import EmbeddingStore
import uvicorn

app = FastAPI(title="Election NLP API")

# Initialize embedding store (assumes index built elsewhere)
emb_store = EmbeddingStore()
try:
    # In a real scenario, load a pre-built index file
    pass  # emb_store.index = faiss.read_index("data/embeddings/index.faiss")
except Exception:
    pass

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/api/search")
def search(request: QueryRequest):
    """
    Search the comment embeddings for nearest neighbors to the query.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
    try:
        indices = emb_store.search(request.query, k=request.top_k)
        return {"indices": indices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

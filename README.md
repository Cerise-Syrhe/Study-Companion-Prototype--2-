# StudyCompanion backend
    pip install -r requirements.txt
    uvicorn app:app --reload        # docs at http://localhost:8000/docs
    python test_core.py             # stdlib-only tests of storage, retrieval, grounding, quiz, learner model

Storage: SQLite file `studyco.db` (WAL-friendly, single file). Retrieval: FTS5 BM25 with Porter stemming.
Optional hybrid search: `Store(path, embed_fn=your_embedding_function)` stores vectors per unit and fuses cosine similarity with BM25.
Endpoints: POST /sources/text, POST /sources/file, GET /sources /topics /graph /units/{id} /search,
POST /ask, POST /quiz, POST /quiz/{id}/answer, GET /users/{id}/mastery, GET /users/{id}/report, PUT /topics/{name}/prerequisites.

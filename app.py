"""FastAPI layer. Run:  uvicorn app:app --reload   (DB file: studyco.db, override with STUDYCO_DB)"""
import os
from typing import List, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel
from core import Store

store = Store(os.environ.get("STUDYCO_DB", "studyco.db"))   # pass embed_fn=... to enable hybrid search
app = FastAPI(title="StudyCompanion API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class AskIn(BaseModel): user: str; query: str
class QuizIn(BaseModel): user: str; topics: Optional[List[str]] = None; n: int = 5; diagnostic: bool = False
class AnswerIn(BaseModel): user: str; given: str
class PrereqIn(BaseModel): prerequisites: List[str]
class TextIn(BaseModel): kind: str = "Notes"; title: str; text: str

@app.post("/sources/text")
def add_text(b: TextIn): return store.add_source(b.kind, b.title, b.text)

@app.post("/sources/file")
async def add_file(kind: str = Form("Notes"), file: UploadFile = File(...)):
    # Plain text/SRT/VTT/Markdown here. Plug video ASR, PDF/PPTX parsing and figure captioning in before add_source.
    text = (await file.read()).decode("utf-8", "ignore")
    return store.add_source(kind, file.filename, text)

@app.get("/sources")
def sources(): return store.sources()
@app.get("/topics")
def topics(): return store.topics()
@app.put("/topics/{name}/prerequisites")
def prereqs(name: str, b: PrereqIn): store.set_prereqs(name, b.prerequisites); return store.graph()
@app.get("/graph")
def graph(): return store.graph()
@app.get("/units/{uid}")
def unit(uid: int):
    u = store.get_unit(uid)
    if not u: raise HTTPException(404, "unit not found")
    return u
@app.get("/search")
def search(q: str, k: int = 3): return store.search(q, k)
@app.post("/ask")
def ask(b: AskIn): return store.ask(b.user, b.query)
@app.post("/quiz")
def quiz(b: QuizIn): return store.make_quiz(b.user, b.topics, b.n, b.diagnostic)
@app.post("/quiz/{qid}/answer")
def answer(qid: int, b: AnswerIn):
    try: return store.answer(b.user, qid, b.given)
    except KeyError: raise HTTPException(404, "question not found for this user")
@app.get("/users/{user}/mastery")
def mastery(user: str): return store.mastery(user)
@app.get("/users/{user}/report")
def report(user: str): return store.report(user)

@app.get("/units")
def units():
    return [dict(r) for r in store.db.execute("SELECT u.id,u.kind,u.loc,t.name topic FROM units u JOIN topics t ON t.id=u.topic_id ORDER BY t.ordinal,u.id")]

@app.get("/", include_in_schema=False)
def home(): return FileResponse(Path(__file__).parent / "static" / "index.html")

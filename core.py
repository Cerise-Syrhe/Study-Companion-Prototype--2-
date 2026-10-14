"""Storage + retrieval + learner model. Pure stdlib (sqlite3 with FTS5)."""
import json, math, random, re, sqlite3, threading, time
from pathlib import Path

STOP = set("a an the of and or to in is are was were be for on with as by that this it its from at which can we use used what how does do why who when while than then them they their there these those not but also into".split())
raw = lambda s: [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 1]
def stem(w): return re.sub(r"(ing|ed|es|s)$", "", w) if len(w) > 3 else w

# Bayesian knowledge tracing parameters
P_INIT, P_GUESS, P_SLIP, P_LEARN = 0.30, 0.20, 0.10, 0.15
GROUNDED, PARTIAL = 0.60, 0.20


class Store:
    def __init__(self, path=":memory:", embed_fn=None):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.embed_fn = embed_fn          # optional: fn(text)->list[float] for hybrid search
        self.db.executescript((Path(__file__).parent / "schema.sql").read_text())

    # ---------- ingestion ----------
    def _topic(self, name, prereqs=None):
        r = self.db.execute("SELECT id FROM topics WHERE name=?", (name,)).fetchone()
        if r: return r["id"]
        n = self.db.execute("SELECT COALESCE(MAX(ordinal),0)+1 FROM topics").fetchone()[0]
        tid = self.db.execute("INSERT INTO topics(name,ordinal) VALUES(?,?)", (name, n)).lastrowid
        if prereqs is None:   # default: depends on the previously introduced topic
            prev = self.db.execute("SELECT id FROM topics WHERE ordinal=?", (n - 1,)).fetchone()
            prereqs = [prev["id"]] if prev else []
        for p in prereqs:
            self.db.execute("INSERT OR IGNORE INTO prereqs VALUES(?,?)", (tid, p))
        return tid

    def set_prereqs(self, topic, prereq_names):
        with self.lock, self.db:
            t = self._topic(topic)
            self.db.execute("DELETE FROM prereqs WHERE topic_id=?", (t,))
            for n in prereq_names:
                self.db.execute("INSERT OR IGNORE INTO prereqs VALUES(?,?)", (t, self._topic(n)))

    def add_source(self, kind, title, text):
        """Lines 'Type|Location|Topic|text' keep exact origin; plain paragraphs get inferred loc/topic."""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        with self.lock, self.db:
            sid = self.db.execute("INSERT INTO sources(kind,title) VALUES(?,?)", (kind, title)).lastrowid
            ids = []
            for i, l in enumerate(lines, 1):
                p = l.split("|")
                if len(p) >= 4: k, loc, topic, body = p[0].strip(), p[1].strip(), p[2].strip(), "|".join(p[3:]).strip()
                else: k, loc, topic, body = kind, f"para {i}", None, l
                if topic is None: topic = self._infer_topic(body)
                fig = int("figure" in loc.lower())
                emb = json.dumps(self.embed_fn(body)).encode() if self.embed_fn else None
                ids.append(self.db.execute(
                    "INSERT INTO units(source_id,topic_id,kind,loc,text,is_figure,embedding) VALUES(?,?,?,?,?,?,?)",
                    (sid, self._topic(topic), k, loc, body, fig, emb)).lastrowid)
            return {"source_id": sid, "units": len(ids)}

    def _infer_topic(self, body):
        ws = sorted({stem(w) for w in raw(body) if len(w) > 4}, key=lambda w: -self._idf(w))
        w = ws[0] if ws else "general"
        return w.capitalize()

    # ---------- retrieval ----------
    def _df(self, term):
        try: return self.db.execute("SELECT COUNT(*) FROM units_fts WHERE units_fts MATCH ?", (f'"{term}"',)).fetchone()[0]
        except sqlite3.OperationalError: return 0
    def _n(self): return self.db.execute("SELECT COUNT(*) FROM units").fetchone()[0]
    def _idf(self, term):
        n, df = self._n(), self._df(term)
        return math.log(1 + (n - df + .5) / (df + .5))

    def search(self, q, k=3):
        terms = list(dict.fromkeys(raw(q)))
        if not terms: return {"hits": [], "coverage": 0.0}
        match = " OR ".join(f'"{t}"' for t in terms)
        rows = self.db.execute(
            """SELECT u.id, u.kind, u.loc, u.text, t.name topic, s.title source, bm25(units_fts) sc
               FROM units_fts JOIN units u ON u.id=units_fts.rowid JOIN topics t ON t.id=u.topic_id
               JOIN sources s ON s.id=u.source_id WHERE units_fts MATCH ? ORDER BY sc LIMIT ?""", (match, k)).fetchall()
        hits = [dict(r, score=-r["sc"]) for r in rows]
        if self.embed_fn and hits is not None:   # hybrid: re-rank with cosine similarity when embeddings exist
            hits = self._hybrid(q, terms, hits, k)
        cov = 0.0
        if hits:
            tot = hit = 0.0
            for t in terms:
                df = self._df(f'"{t}"'.strip('"'))
                w = self._idf(t) if df else math.log(1 + self._n())   # unknown words count as maximally rare
                tot += w
                ok = self.db.execute("SELECT 1 FROM units_fts WHERE units_fts MATCH ? AND rowid=?", (f'"{t}"', hits[0]["id"])).fetchone()
                hit += w if ok else 0
            cov = hit / tot if tot else 0.0
        return {"hits": hits, "coverage": cov}

    def _hybrid(self, q, terms, hits, k):
        qv = self.embed_fn(q)
        cos = lambda a, b: sum(x * y for x, y in zip(a, b)) / ((sum(x*x for x in a) ** .5) * (sum(y*y for y in b) ** .5) or 1)
        pool = {h["id"]: h for h in hits}
        for r in self.db.execute("SELECT u.id,u.kind,u.loc,u.text,t.name topic,s.title source,u.embedding FROM units u JOIN topics t ON t.id=u.topic_id JOIN sources s ON s.id=u.source_id WHERE embedding IS NOT NULL"):
            h = pool.setdefault(r["id"], {k2: r[k2] for k2 in ("id","kind","loc","text","topic","source")} | {"score": 0.0})
            h["sim"] = cos(qv, json.loads(r["embedding"]))
        mx = max((h["score"] for h in pool.values()), default=1) or 1
        for h in pool.values(): h["fused"] = .5 * h["score"] / mx + .5 * h.get("sim", 0)
        return sorted(pool.values(), key=lambda h: -h["fused"])[:k]

    def ask(self, user, q):
        r = self.search(q)
        cov = r["coverage"]
        status = "declined" if not r["hits"] or cov < PARTIAL else "partial" if cov < GROUNDED else "grounded"
        ctx = [] if status == "declined" else r["hits"][:2]
        cites = [{"unit_id": h["id"], "source": h["source"], "kind": h["kind"], "loc": h["loc"],
                  "topic": h["topic"], "excerpt": h["text"]} for h in ctx]
        note = {"declined": "Not covered in the uploaded material.",
                "partial": "Only partly covered; anything beyond the excerpts is outside the sources.",
                "grounded": "Answered from the uploaded material."}[status]
        with self.lock, self.db:
            self.db.execute("INSERT INTO chat_log(user_id,query,status,unit_ids) VALUES(?,?,?,?)", (user, q, status, json.dumps([c["unit_id"] for c in cites])))
        return {"status": status, "coverage": round(cov, 3), "note": note, "citations": cites}

    def get_unit(self, uid):
        r = self.db.execute("SELECT u.*, t.name topic, s.title source FROM units u JOIN topics t ON t.id=u.topic_id JOIN sources s ON s.id=u.source_id WHERE u.id=?", (uid,)).fetchone()
        if not r: return None
        d = dict(r); d.pop("embedding", None); return d

    # ---------- learner model ----------
    def ensure_user(self, user):
        with self.lock, self.db:
            self.db.execute("INSERT OR IGNORE INTO users(id) VALUES(?)", (user,))
            for t in self.db.execute("SELECT id FROM topics"):
                self.db.execute("INSERT OR IGNORE INTO mastery(user_id,topic_id,p,n,updated) VALUES(?,?,?,0,?)", (user, t["id"], P_INIT, time.time()))

    def mastery(self, user):
        self.ensure_user(user)
        return {r["name"]: {"p": round(r["p"], 3), "answers": r["n"]} for r in self.db.execute(
            "SELECT t.name,m.p,m.n FROM mastery m JOIN topics t ON t.id=m.topic_id WHERE m.user_id=? ORDER BY t.ordinal", (user,))}

    @staticmethod
    def bkt(p, correct):
        post = p*(1-P_SLIP)/(p*(1-P_SLIP)+(1-p)*P_GUESS) if correct else p*P_SLIP/(p*P_SLIP+(1-p)*(1-P_GUESS))
        return post + (1 - post) * P_LEARN

    # ---------- assessment ----------
    def _terms(self, text):
        seen = {}
        for w in raw(text):
            if len(w) > 4: seen.setdefault(stem(w), w)
        return sorted(seen.values(), key=lambda w: -self._idf(stem(w)))

    def make_quiz(self, user, topics=None, n=5, diagnostic=False):
        self.ensure_user(user)
        q = "SELECT u.id,u.text,u.topic_id,t.name topic,m.p FROM units u JOIN topics t ON t.id=u.topic_id JOIN mastery m ON m.topic_id=u.topic_id AND m.user_id=?"
        args = [user]
        if topics:
            q += " WHERE t.name IN (%s)" % ",".join("?" * len(topics)); args += topics
        pool = [dict(r) for r in self.db.execute(q, args)]
        random.shuffle(pool)
        if diagnostic:
            seen, one = set(), []
            for u in pool:
                if u["topic_id"] not in seen: seen.add(u["topic_id"]); one.append(u)
            pool, n = one, len(one)
        pool = sorted(pool, key=lambda u: u["p"] + random.uniform(-.1, .1))[:n]
        out = []
        with self.lock, self.db:
            for i, u in enumerate(pool):
                terms = self._terms(u["text"])
                if not terms: continue
                fresh = [w for w in terms if not self.db.execute("SELECT 1 FROM asked WHERE user_id=? AND unit_id=? AND answer=?", (user, u["id"], w)).fetchone()]
                ans, repeated = (fresh[0], False) if fresh else (terms[0], True)
                self.db.execute("INSERT OR IGNORE INTO asked VALUES(?,?,?)", (user, u["id"], ans))
                df = self._df(stem(ans))
                diff = "hard" if df <= 1 else "medium" if df == 2 else "easy"
                prompt = re.sub(rf"\b{re.escape(ans)}\b", "_____", u["text"], count=1, flags=re.I)
                qtype, opts = ("mcq" if i % 2 == 0 else "short"), None
                if qtype == "mcq":
                    others = {w for r in self.db.execute("SELECT text FROM units WHERE id!=?", (u["id"],)) for w in self._terms(r["text"]) if stem(w) != stem(ans)}
                    opts = random.sample(sorted(others), min(3, len(others))) + [ans]; random.shuffle(opts)
                qid = self.db.execute("INSERT INTO questions(user_id,unit_id,topic_id,qtype,difficulty,prompt,options,answer) VALUES(?,?,?,?,?,?,?,?)",
                                      (user, u["id"], u["topic_id"], qtype, diff, prompt, json.dumps(opts), ans)).lastrowid
                out.append({"question_id": qid, "type": qtype, "topic": u["topic"], "difficulty": diff, "prompt": prompt,
                            "options": opts, "source_unit_id": u["id"], "repeated": repeated})
        return out

    def answer(self, user, qid, given):
        q = self.db.execute("SELECT * FROM questions WHERE id=? AND user_id=?", (qid, user)).fetchone()
        if not q: raise KeyError("unknown question")
        ok = stem(given.strip().lower()) == stem(q["answer"].lower())
        with self.lock, self.db:
            self.ensure_user(user)
            m = self.db.execute("SELECT p FROM mastery WHERE user_id=? AND topic_id=?", (user, q["topic_id"])).fetchone()
            newp = self.bkt(m["p"], ok)
            self.db.execute("UPDATE mastery SET p=?, n=n+1, updated=? WHERE user_id=? AND topic_id=?", (newp, time.time(), user, q["topic_id"]))
            self.db.execute("INSERT INTO attempts(user_id,question_id,given,correct) VALUES(?,?,?,?)", (user, qid, given, int(ok)))
        u = self.get_unit(q["unit_id"])
        return {"correct": ok, "answer": q["answer"], "mastery_after": round(newp, 3),
                "citation": {"unit_id": u["id"], "source": u["source"], "kind": u["kind"], "loc": u["loc"], "excerpt": u["text"]}}

    def report(self, user, last=None):
        mast = self.mastery(user)
        rows = self.db.execute("""SELECT t.name topic, q.answer, a.given, a.correct, q.unit_id FROM attempts a
            JOIN questions q ON q.id=a.question_id JOIN topics t ON t.id=q.topic_id WHERE a.user_id=? ORDER BY a.id DESC LIMIT ?""", (user, last or 1000)).fetchall()
        wrong = [r for r in rows if not r["correct"]]
        return {"mastery": mast, "answered": len(rows), "accuracy": round(1 - len(wrong) / len(rows), 3) if rows else None,
                "weak_topics": [t for t, v in mast.items() if v["p"] < .5],
                "misconceptions": [{"topic": r["topic"], "said": r["given"], "source_says": r["answer"], "unit_id": r["unit_id"]} for r in wrong[:20]]}

    def graph(self):
        return {"nodes": [dict(r) for r in self.db.execute("SELECT id,name FROM topics ORDER BY ordinal")],
                "edges": [{"from": r["prereq_id"], "to": r["topic_id"]} for r in self.db.execute("SELECT * FROM prereqs")]}

    def topics(self):
        return [dict(r) for r in self.db.execute("SELECT t.id,t.name,COUNT(u.id) units FROM topics t LEFT JOIN units u ON u.topic_id=t.id GROUP BY t.id ORDER BY t.ordinal")]

    def sources(self):
        return [dict(r) for r in self.db.execute("SELECT s.*,COUNT(u.id) units FROM sources s LEFT JOIN units u ON u.source_id=s.id GROUP BY s.id")]

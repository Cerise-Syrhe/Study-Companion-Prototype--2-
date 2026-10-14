PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, created REAL DEFAULT (strftime('%s','now')));
CREATE TABLE IF NOT EXISTS sources(id INTEGER PRIMARY KEY, kind TEXT, title TEXT, created REAL DEFAULT (strftime('%s','now')));
CREATE TABLE IF NOT EXISTS topics(id INTEGER PRIMARY KEY, name TEXT UNIQUE, ordinal INTEGER);
CREATE TABLE IF NOT EXISTS prereqs(topic_id INTEGER REFERENCES topics(id), prereq_id INTEGER REFERENCES topics(id), PRIMARY KEY(topic_id,prereq_id));
CREATE TABLE IF NOT EXISTS units(
  id INTEGER PRIMARY KEY, source_id INTEGER REFERENCES sources(id) ON DELETE CASCADE,
  topic_id INTEGER REFERENCES topics(id), kind TEXT, loc TEXT, text TEXT NOT NULL,
  is_figure INTEGER DEFAULT 0, embedding BLOB);
CREATE VIRTUAL TABLE IF NOT EXISTS units_fts USING fts5(text, content='units', content_rowid='id', tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS u_ai AFTER INSERT ON units BEGIN INSERT INTO units_fts(rowid,text) VALUES (new.id,new.text); END;
CREATE TRIGGER IF NOT EXISTS u_ad AFTER DELETE ON units BEGIN INSERT INTO units_fts(units_fts,rowid,text) VALUES('delete',old.id,old.text); END;
CREATE TABLE IF NOT EXISTS mastery(user_id TEXT REFERENCES users(id), topic_id INTEGER REFERENCES topics(id), p REAL, n INTEGER DEFAULT 0, updated REAL, PRIMARY KEY(user_id,topic_id));
CREATE TABLE IF NOT EXISTS questions(
  id INTEGER PRIMARY KEY, user_id TEXT, unit_id INTEGER REFERENCES units(id), topic_id INTEGER,
  qtype TEXT, difficulty TEXT, prompt TEXT, options TEXT, answer TEXT, created REAL DEFAULT (strftime('%s','now')));
CREATE TABLE IF NOT EXISTS asked(user_id TEXT, unit_id INTEGER, answer TEXT, PRIMARY KEY(user_id,unit_id,answer));
CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY, user_id TEXT, question_id INTEGER REFERENCES questions(id), given TEXT, correct INTEGER, ts REAL DEFAULT (strftime('%s','now')));
CREATE TABLE IF NOT EXISTS chat_log(id INTEGER PRIMARY KEY, user_id TEXT, query TEXT, status TEXT, unit_ids TEXT, ts REAL DEFAULT (strftime('%s','now')));
CREATE INDEX IF NOT EXISTS ix_units_topic ON units(topic_id);
CREATE INDEX IF NOT EXISTS ix_att_user ON attempts(user_id);

import random, re
from core import Store
SAMPLE = """Slides|Slide 3|Supervised Learning|Supervised learning trains a model on labeled examples, mapping inputs to known outputs so it can predict labels for new data.
Textbook|p. 12|Supervised Learning|Classification predicts discrete categories while regression predicts continuous numeric values; both are supervised tasks.
Video|04:10|Linear Regression|Linear regression fits a line to data by minimizing the sum of squared residuals between predictions and targets.
Slides|Slide 8|Linear Regression|The normal equation gives the least squares weights directly, but it becomes slow when there are many features.
Video|12:45|Gradient Descent|Gradient descent iteratively updates parameters in the direction opposite the gradient of the loss to reduce error.
Textbook|p. 47|Gradient Descent|The learning rate controls step size; a rate that is too large causes divergence while one too small makes convergence slow.
Slides|Slide 9 (figure)|Gradient Descent|Figure caption extracted from image: contour plot of a loss surface showing successive gradient descent steps moving toward the minimum.
Slides|Slide 15|Logistic Regression|Logistic regression applies the sigmoid function to a linear score to output a probability between zero and one for classification.
Textbook|p. 61|Logistic Regression|Cross entropy loss penalizes confident wrong predictions and is the standard objective for logistic regression.
Video|21:30|Overfitting|Overfitting happens when a model memorizes noise in the training set, giving low training error but high validation error.
Slides|Slide 22|Overfitting|Cross validation splits data into folds to estimate generalization performance and detect overfitting early.
Textbook|p. 88|Regularization|Regularization adds a penalty on weight magnitude to the loss, discouraging complex models and reducing variance.
Video|30:05|Regularization|Lasso uses an L1 penalty that drives some weights exactly to zero, performing feature selection, while ridge uses an L2 penalty that shrinks weights smoothly."""

def fresh():
    s = Store(); r = s.add_source("Course", "ML 101", SAMPLE); assert r["units"] == 13
    s.set_prereqs("Gradient Descent", ["Linear Regression"]); s.set_prereqs("Logistic Regression", ["Supervised Learning", "Gradient Descent"])
    return s

def test_retrieval_and_grounding():
    s = fresh()
    a = s.ask("u1", "effect of too large learning rate")
    assert a["status"] == "grounded" and a["citations"][0]["loc"] == "p. 47", a
    a = s.ask("u1", "how does linear regression fit a line")
    assert a["citations"][0]["loc"] == "04:10"
    for q in ["who won the cricket world cup", "how does a blockchain work", "what is the capital of France"]:
        assert s.ask("u1", q)["status"] == "declined", q
    assert s.ask("u1", "what is the model of tesla")["status"] in ("partial", "declined")
    assert s.get_unit(8)["loc"] == "Slide 15"

def test_learner_model_and_quiz():
    s = fresh(); u = "stu"
    assert all(v["p"] == .3 for v in s.mastery(u).values())
    diag = s.make_quiz(u, diagnostic=True); assert len(diag) == len(s.topics())
    for q in diag:
        assert "_____" in q["prompt"] and (q["type"] == "short" or len(q["options"]) >= 2)
        r = s.answer(u, q["question_id"], "wrong-guess"); assert not r["correct"] and r["citation"]["loc"]
    m = s.mastery(u); assert all(v["p"] < .3 for v in m.values())
    q = s.make_quiz(u, ["Overfitting"], 2)[0]
    ans = s.db.execute("SELECT answer FROM questions WHERE id=?", (q["question_id"],)).fetchone()[0]
    before = s.mastery(u)["Overfitting"]["p"]; assert s.answer(u, q["question_id"], ans)["correct"]
    assert s.mastery(u)["Overfitting"]["p"] > before
    rep = s.report(u); assert rep["weak_topics"] and rep["misconceptions"]
    assert len(s.graph()["edges"]) >= 5

def test_no_repeats_and_isolation():
    s = fresh(); seen = set()
    for _ in range(4):
        for q in s.make_quiz("a", n=5):
            seen_key = (q["source_unit_id"], re.sub(r"\s+", " ", q["prompt"])); assert seen_key not in seen; seen.add(seen_key)
    assert s.mastery("b")["Overfitting"]["answers"] == 0

def test_simulated_student_gain():
    random.seed(1); s = fresh(); u = "sim"; start = sum(v["p"] for v in s.mastery(u).values())
    for _ in range(5):
        for q in s.make_quiz(u, n=5):
            ans = s.db.execute("SELECT answer FROM questions WHERE id=?", (q["question_id"],)).fetchone()[0]
            s.answer(u, q["question_id"], ans if random.random() < .9 else "x")
    assert sum(v["p"] for v in s.mastery(u).values()) > start

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)

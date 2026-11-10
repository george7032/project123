import io, json, os, re
from collections import Counter

import anthropic
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score

st.set_page_config(page_title="CBC Study Tutor", page_icon="📘")

# KICD designs number strands "1.0 NUMBERS" and sub-strands "1.1 Whole Numbers".
# Check the parsed output for your PDFs and adjust these patterns if needed.
STRAND = re.compile(r"^\s*(\d+)\.0\s+([A-Z][A-Za-z ,&'-]{3,60})\s*$")
SUB = re.compile(r"^\s*(\d+\.\d+)\s+([A-Z][A-Za-z ,&'-]{3,70})\s*$")

GEN = """You are a Kenyan CBC {subject} teacher for Grade {grade}.
Using ONLY the curriculum extract below, write {n} practice questions on "{strand}".
Mix recall with application questions set in everyday Kenyan contexts (market, farm, M-Pesa, matatu).
Return JSON only: {{"questions":[{{"question":"","answer":"","explanation":"","outcome":"the learning outcome it assesses"}}]}}
EXTRACT:
{ctx}"""

FB = """Assess this Grade {grade} {subject} answer using CBC levels: Exceeds, Meets, Approaches, Below Expectations.
Question: {q}
Model answer: {a}
Student answer: {s}
Return JSON only: {{"level":"","feedback":"2-3 encouraging sentences","next_step":"one thing to practise"}}"""


@st.cache_data(show_spinner="Parsing PDFs...")
def parse(files):
    chunks = []
    for name, data in files:
        strand, sub, buf, lines = "General", "General", [], []
        for p in PdfReader(io.BytesIO(data)).pages:
            lines += (p.extract_text() or "").splitlines()
        for ln in lines + ["9.9 FINAL"]:
            m1, m2 = STRAND.match(ln), SUB.match(ln)
            if m1 or m2:
                text = re.sub(r"\s+", " ", " ".join(buf)).strip()
                for i in range(0, len(text), 900):
                    piece = text[i:i + 1000]
                    if len(piece) > 150:
                        chunks.append({"file": name, "strand": strand, "sub": sub, "text": piece})
                buf = []
                if m1:
                    strand = sub = m1.group(2).title()
                else:
                    sub = m2.group(2).title()
            else:
                buf.append(ln)
    return chunks


def build(chunks):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform([c["text"] for c in chunks])
    y = [c["strand"] for c in chunks]
    clf = f1 = None
    if len(set(y)) > 1 and min(Counter(y).values()) >= 3:
        clf = LogisticRegression(max_iter=1000).fit(X, y)
        f1 = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=3, scoring="f1_macro").mean()
    return {"chunks": chunks, "vec": vec, "X": X, "clf": clf, "f1": f1}


def retrieve(idx, strand, query, k=4):
    ids = [i for i, c in enumerate(idx["chunks"]) if c["strand"] == strand]
    sims = cosine_similarity(idx["vec"].transform([query]), idx["X"][ids]).ravel()
    return [idx["chunks"][ids[i]] for i in sims.argsort()[::-1][:k]]


def ask(prompt, model, key, max_tokens=2000):
    r = anthropic.Anthropic(api_key=key).messages.create(
        model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
    return r.content[0].text


def as_json(t):
    return json.loads(t[t.find("{"): t.rfind("}") + 1])


ss = st.session_state
ss.setdefault("idx", None)
ss.setdefault("qs", [])
ss.setdefault("stats", [0, 0])

st.title("📘 CBC Study Tutor")

with st.sidebar:
    key = st.text_input("Anthropic API key", os.getenv("ANTHROPIC_API_KEY", ""), type="password")
    model = st.text_input("Model", "claude-sonnet-5-5")
    subject = st.text_input("Subject", "Mathematics")
    grade = st.selectbox("Grade", list(range(4, 10)), index=3)
    files = st.file_uploader("KICD curriculum design PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Build index") and files:
        chunks = parse([(f.name, f.getvalue()) for f in files])
        if chunks:
            ss.idx = build(chunks)
        else:
            st.error("No sections detected. Check the PDF or the heading patterns in app.py.")
    if ss.idx:
        st.success(f"{len(ss.idx['chunks'])} chunks, {len({c['strand'] for c in ss.idx['chunks']})} strands")
        if ss.idx["f1"] is not None:
            st.metric("Strand classifier macro-F1 (3-fold CV)", f"{ss.idx['f1']:.2f}")

if not ss.idx:
    st.info("Upload a KICD curriculum design PDF in the sidebar, then click Build index.")
    st.stop()

strand = st.selectbox("Strand", sorted({c["strand"] for c in ss.idx["chunks"]}))
topic = st.text_input("Focus (optional)")
n = st.slider("Questions", 1, 8, 4)
only = st.checkbox("Show only curriculum-aligned questions", True)

if st.button("Generate questions"):
    if not key:
        st.error("Add an API key in the sidebar.")
        st.stop()
    ctx = "\n---\n".join(c["text"] for c in retrieve(ss.idx, strand, topic or strand))
    try:
        qs = as_json(ask(GEN.format(subject=subject, grade=grade, n=n, strand=strand, ctx=ctx), model, key))["questions"]
    except Exception as e:
        st.error(f"Generation failed: {e}")
        st.stop()
    clf, vec = ss.idx["clf"], ss.idx["vec"]
    for q in qs:
        if clf:
            q["pred"] = clf.predict(vec.transform([q["question"] + " " + q["answer"]]))[0]
            q["ok"] = q["pred"] == strand
        else:
            q["pred"], q["ok"] = None, True
    ss.qs = qs
    ss.stats[0] += len(qs)
    ss.stats[1] += sum(q["ok"] for q in qs)

if ss.stats[0]:
    st.metric("Alignment pass rate (all sessions)", f"{ss.stats[1] / ss.stats[0]:.0%}")

for i, q in enumerate(ss.qs):
    if only and not q["ok"]:
        continue
    st.markdown(f"**Q{i + 1}. {q['question']}**")
    st.caption(f"Outcome: {q.get('outcome', '')} · classifier: {q['pred']} {'✅' if q['ok'] else '⚠️'}")
    s = st.text_area("Your answer", key=f"a{i}")
    c1, c2 = st.columns(2)
    if c1.button("Get feedback", key=f"f{i}") and s.strip():
        try:
            fb = as_json(ask(FB.format(grade=grade, subject=subject, q=q["question"], a=q["answer"], s=s), model, key, 600))
            st.info(f"**{fb['level']}**: {fb['feedback']}\n\n*Next:* {fb['next_step']}")
        except Exception as e:
            st.error(f"Feedback failed: {e}")
    with c2.expander("Model answer"):
        st.write(q["answer"])
        st.write(q["explanation"])

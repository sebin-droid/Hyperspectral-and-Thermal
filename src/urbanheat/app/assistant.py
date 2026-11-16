"""Ask-the-map assistant: retrieval over the project's own documents (RAG) + Claude for the wording.

The scientific numbers always come from the model and the XAI (passed in as `facts`); the language model only
explains them, grounded in retrieved passages. Without Claude credentials (ANTHROPIC_API_KEY, or an
`ant auth login` profile) the assistant still answers offline from the same facts and passages.
"""
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DOCS = Path(__file__).resolve().parents[3] / "docs"
MODEL = "claude-opus-5-5"
SYSTEM = (
    "You explain urban heat maps to city planners and students. Answer from the FACTS (model outputs for the "
    "place the user selected) and the numbered PASSAGES only. Cite passages as [1], [2]. Give numbers with units "
    "(deg C). Say plainly that predictions are land surface temperature, not air temperature, and that "
    "attributions and scenarios are model estimates, not proof of cause. If the facts and passages do not answer "
    "the question, say so. Keep it under 200 words, in plain language."
)


def chunks(paths=None):
    """[(source, title, text)]: one passage per '##' section of each markdown file in docs/."""
    out = []
    for p in paths or sorted(DOCS.glob("*.md")):
        parts = re.split(r"^#{2,3} +", Path(p).read_text(encoding="utf-8"), flags=re.M)
        for part in parts[1:]:
            title, _, body = part.partition("\n")
            if body.strip():
                out.append((Path(p).name, title.strip(), " ".join(body.split())))
    return out


class Knowledge:
    """TF-IDF index over the passages (no embedding service needed; the corpus is small and domain-specific)."""

    def __init__(self, passages=None):
        self.passages = passages or chunks()
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.m = self.vec.fit_transform([f"{t} {t} {x}" for _, t, x in self.passages])   # title counts twice

    def search(self, query, k=4):
        s = (self.m @ self.vec.transform([query]).T).toarray()[:, 0]
        best = np.argsort(-s)[:k]
        return [dict(source=self.passages[i][0], title=self.passages[i][1], text=self.passages[i][2],
                     score=float(s[i])) for i in best if s[i] > 0]


def format_facts(facts):
    return "\n".join(f"- {k}: {v}" for k, v in facts.items()) or "- (no place selected)"


def offline_answer(question, facts, hits):
    """Answer built from the facts and the best passage, used when Claude is unavailable."""
    lines = []
    if "top_factors" in facts:
        lines.append(f"The model's main factors here: {facts['top_factors']}.")
    if "predicted_anomaly_c" in facts:
        lines.append(f"Predicted LST anomaly: {facts['predicted_anomaly_c']} (relative to the city median).")
    if "scenario" in facts:
        lines.append(f"Scenario: {facts['scenario']}.")
    if hits:
        lines.append(f"Background [1] {hits[0]['title']}: {hits[0]['text'][:400]}")
    lines.append("These are modelled land surface temperatures, not air temperatures or proof of cause.")
    return " ".join(lines)


def ask(question, facts=None, knowledge=None, client=None, k=4):
    """{answer, sources, mode}: mode is 'claude' or 'offline' (no credentials, network or API failure)."""
    import anthropic

    facts, knowledge = facts or {}, knowledge or Knowledge()
    hits = knowledge.search(f"{question} {question} {facts.get('top_factors', '')} {facts.get('scenario', '')}", k)
    passages = "\n\n".join(f"[{i}] {h['title']} ({h['source']}): {h['text']}" for i, h in enumerate(hits, 1))
    prompt = f"FACTS\n{format_facts(facts)}\n\nPASSAGES\n{passages}\n\nQUESTION\n{question}"
    sources = [dict(n=i, title=h["title"], source=h["source"]) for i, h in enumerate(hits, 1)]
    try:
        client = client or anthropic.Anthropic()
        r = client.beta.messages.create(
            model=MODEL, max_tokens=16000, system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",   # re-runs on a fallback model if declined
        )
    except (TypeError, anthropic.CredentialsError, anthropic.AuthenticationError, anthropic.PermissionDeniedError,
            anthropic.APIConnectionError) as e:                    # TypeError / CredentialsError: no credentials set
        return dict(answer=offline_answer(question, facts, hits), sources=sources, mode="offline", error=str(e)[:200])
    except anthropic.APIStatusError as e:
        return dict(answer=offline_answer(question, facts, hits), sources=sources, mode="offline",
                    error=f"Claude API error {e.status_code}")
    if r.stop_reason == "refusal":
        return dict(answer=offline_answer(question, facts, hits), sources=sources, mode="offline", error="declined")
    text = "".join(b.text for b in r.content if b.type == "text")
    return dict(answer=text, sources=sources, mode="claude")

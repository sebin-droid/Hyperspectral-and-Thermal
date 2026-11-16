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
    "the question, say so. Write for people with no science background: answer the question in the first sentence, "
    "use short sentences and everyday words, and explain any technical term in a few words. Keep it under 150 words."
)


def chunks(paths=None):
    """[(source, title, text)]: one passage per '##' section of each markdown file (default: docs/KNOWLEDGE.md, the
    text written for dashboard users; the team docs are developer jargon and mislead the answers)."""
    out = []
    for p in paths or [DOCS / "KNOWLEDGE.md"]:
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


CAVEAT_SHORT = "(This is the temperature of the ground seen from space, not the air; the model gives estimates, not proof.)"
HERE = ("this spot", "this place", "this area", "here", "this circle", "this city", "parts of")
INTENTS = [   # first match wins
    ("accuracy", ("sure", "accura", "trust", "reliab", "error", "confiden", "correct")),
    ("whatif", ("tree", "plant", "park", "greener", "what if", "reduce", "cool it", "cool down", "help", "mitigat")),
    ("define", ("what is", "what does", "what are", "meaning", "mean?", "define", "air temperature", "air temp", "feel",
                "thermometer", "ndvi", "ndbi", "mndwi", "built share", "heat island", "hyperspectral", "satellite",
                "how does the model", "how does it work", "other days")),
    ("why", ("why", "reason", "cause", "factor", "because")),
    ("hotcool", ("hot", "cool", "cold", "warm", "temperature", "heat")),
]


def intent(question):
    """Kind of question. "Is this spot hot?" / "which parts are hottest?" ask about the map, not a definition."""
    q = question.lower()
    if any(w in q for w in ("which part", "which area", "where", "parts of", "hottest part", "coolest part")):
        return "where"
    if any(h in q for h in HERE) and any(w in q for w in ("hot", "cool", "cold", "warm")) and "why" not in q:
        return "hotcool"
    return next((name for name, words in INTENTS if any(w in q for w in words)), "other")


def plain(text):
    """The plain-language part of a KNOWLEDGE.md passage (everything before 'Details:')."""
    return text.split(" Details:")[0].strip()


def offline_answer(question, facts, knowledge):
    """(answer, titles of the passages used): a plain answer to the kind of question asked, from the facts (place
    card or clicked spot) and the plain part of the best passage. Used when Claude is unavailable."""
    q, kind = question.lower(), intent(question)
    hit = (knowledge.search(question, 1) or [None])[0]
    used = []
    spot = "measured_vs_typical_c" in facts
    place = facts.get("place") or ", ".join(x for x in (facts.get("name"), facts.get("region")) if x) or "this place"
    reasons = facts.get("top_factors", "")

    def passage(query):
        h = knowledge.search(query, 1)
        if not h:
            return ""
        used.append(h[0]["title"])
        return plain(h[0]["text"])

    if kind == "define" and hit:
        used.append(hit["title"])
        out = plain(hit["text"])
    elif kind == "hotcool" and spot:
        d, typ, here = facts["measured_vs_typical_c"], facts["city_typical_surface_c"], facts["measured_surface_c"]
        asks_cool = any(w in q for w in ("cool", "cold"))
        if abs(d) < 0.3:
            out = f"Neither: the ground in this circle is {here} °C, about the same as the typical {typ} °C of {place}."
        else:
            yes = (d < 0) == asks_cool
            out = (f"{'Yes' if yes else 'No'}. The ground in this circle is {here} °C, {abs(d):.1f} °C "
                   f"{'cooler' if d < 0 else 'hotter'} than the typical {typ} °C of {place}.")
        if reasons:
            out += f" The biggest reasons, according to the model: {reasons}."
    elif kind in ("hotcool", "where") and "typical_c" in facts:
        out = (f"In {place} the ground was typically {facts['typical_c']} °C on {facts['date']}. The hottest parts "
               f"reached about {facts['hottest_c']} °C and the coolest about {facts['coolest_c']} °C. On the map, choose "
               f"'Hot and cool spots': red areas are the hottest, blue the coolest. Press 'Hottest spot' to see one up close.")
    elif kind == "why" and spot:
        out = (f"The model estimates this circle at {facts['model_vs_typical_c']:+.1f} °C compared with the rest of "
               f"{place}. Starting from an average place ({facts['average_place_c']:+.1f} °C), the biggest reasons are: "
               f"{reasons}. 'Heat seen on other days' means this spot was also hot (or cool) in other satellite pictures.")
    elif kind == "accuracy" and "model_error_c" in facts:
        out = (f"Here the model is off by {facts['model_error_c']} °C on average"
               f"{', on a picture it never saw while learning' if facts.get('tested') else ''}. "
               + passage("how accurate is the model"))
    elif kind == "accuracy" and spot:
        out = (f"For this circle the model says {facts['model_vs_typical_c']:+.1f} °C and the satellite measured "
               f"{facts['measured_vs_typical_c']:+.1f} °C (both compared with the city's typical value). "
               + passage("how accurate is the model"))
    elif kind == "whatif":
        out = (("Use step 4, 'What if this area changed?', to see the model's estimate for this circle. " if spot else
                "Click a spot on the map, then use step 4, 'What if this area changed?'. ")
               + passage("how can a hot place be cooled trees") + " " + passage("can I trust the what if results"))
    elif kind == "why":
        out = "Click a spot on the map first (or press 'Hottest spot'); the card then shows the reasons for that spot. "
        out += passage("what makes places hot")
    elif hit:
        used.append(hit["title"])
        out = plain(hit["text"])
    else:
        out = "Sorry, I don't know that. Try one of the suggested questions."
    return f"{out} {CAVEAT_SHORT}", used


def ask(question, facts=None, knowledge=None, client=None, k=4):
    """{answer, sources, mode}: mode is 'claude' or 'offline' (no credentials, network or API failure)."""
    import anthropic

    facts, knowledge = facts or {}, knowledge or Knowledge()
    hits = knowledge.search(f"{question} {question} {facts.get('top_factors', '')} {facts.get('scenario', '')}", k)
    passages = "\n\n".join(f"[{i}] {h['title']} ({h['source']}): {h['text']}" for i, h in enumerate(hits, 1))
    prompt = f"FACTS\n{format_facts(facts)}\n\nPASSAGES\n{passages}\n\nQUESTION\n{question}"
    sources = [dict(n=i, title=h["title"], source=h["source"]) for i, h in enumerate(hits, 1)]

    def offline(error):
        text, used = offline_answer(question, facts, knowledge)
        return dict(answer=text, sources=[dict(n=i, title=t, source="KNOWLEDGE.md") for i, t in enumerate(used, 1)],
                    mode="offline", error=error)

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
        return offline(str(e)[:200])
    except anthropic.APIStatusError as e:
        return offline(f"Claude API error {e.status_code}")
    if r.stop_reason == "refusal":
        return offline("declined")
    text = "".join(b.text for b in r.content if b.type == "text")
    return dict(answer=text, sources=sources, mode="claude")

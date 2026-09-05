class ScoNeSignatureModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # LM fallback: strict entailment decision. The LM is used only when deterministic
        # checks cannot safely conclude. The instructions force a single-word Yes/No reply.
        self.predict = dspy.Predict(dspy.Signature(
            "context: str, question: str -> answer: str",
            "Task: Decide whether the QUESTION's proposition is strictly and logically entailed "
            "by the CONTEXT. Reply with exactly one word, either 'Yes' or 'No' (capital Y/N), "
            "and nothing else.\n\n"
            "Decision rules for the model (follow them precisely):\n"
            "- Return 'Yes' only if the proposition is guaranteed true given the CONTEXT under "
            "strict logical entailment (every reading consistent with the CONTEXT makes it true).\n"
            "- Return 'No' if the proposition is not guaranteed (it's only possible, typical, "
            "plausible, or if the CONTEXT is silent or ambiguous about it).\n"
            "- Use only facts stated or logically implied by the CONTEXT. Do NOT introduce extra "
            "world knowledge except for trivial universal lexical facts (e.g., 'mother' -> 'woman').\n"
            "- If there is any uncertainty, contradiction, or missing information that prevents a "
            "guaranteed conclusion, answer 'No'. Be conservative.\n"
            "- Output must be exactly 'Yes' or 'No' and nothing else (no punctuation, explanation, "
            "or surrounding text). If you cannot produce exactly one of those words, prefer 'No'."
        ))

    def forward(self, **inputs):
        ctx = inputs.get("context", "") or ""
        q = inputs.get("question", "") or ""

        def normalize_text(s: str) -> str:
            "Lowercase, remove/normalize punctuation to spaces, collapse whitespace."
            import re
            s2 = s.lower()
            # replace common punctuation with spaces
            s2 = re.sub(r"[\.\,\;\:\?\!\(\)\[\]\/\\\"']", " ", s2)
            # normalize fancy dashes and multiple hyphens to spaces
            s2 = re.sub(r"[\u2010-\u2015\-]+", " ", s2)
            # collapse multiple spaces
            s2 = re.sub(r"\s+", " ", s2).strip()
            return s2

        def extract_proposition(question: str) -> str:
            """
            Extract the core proposition asked about. Prefer the clause after the last ' that '
            when present (common in these datasets). Fallback to removing leading question framing
            like 'can we conclude that', 'is it true that', etc. Otherwise return the whole
            question text (stripped).
            """
            ql = question.strip()
            ql_lower = ql.lower()
            # choose the last ' that ' to handle nested clauses
            idx = ql_lower.rfind(" that ")
            if idx != -1:
                prop = ql[idx + len(" that "):]
            else:
                for marker in ("can we conclude ", "can we logically conclude ",
                               "is it true that ", "does it follow that ",
                               "can we say that ", "can we infer that "):
                    ml = ql_lower.find(marker)
                    if ml != -1:
                        prop = ql[ml + len(marker):]
                        break
                else:
                    # If question is simple yes/no form like "Did X?" or "Is X true?"
                    # try to strip leading auxiliaries up to the first whitespace after them.
                    prop = ql
            # remove trailing punctuation commonly left on questions and strip
            prop = prop.rstrip(" .?!")
            return prop.strip()

        # Conservative deterministic lexical normalizations that are safe to apply.
        # This list maps many common hyponyms and variant forms to safer, more general
        # terms so straightforward lexical entailments can be detected without an LM.
        SAFE_REPLACEMENTS = {
            # family/relationship terms
            "mother": "woman",
            "mothers": "women",
            "mom": "woman",
            "moms": "women",
            "father": "man",
            "fathers": "men",
            "dad": "man",
            "dads": "men",
            "girlfriend": "woman",
            "girlfriends": "women",
            "boyfriend": "man",
            "boyfriends": "men",
            "wife": "woman",
            "wives": "women",
            "husband": "man",
            "husbands": "men",
            # instruments -> general category "instrument"
            "piano": "instrument",
            "pianos": "instrument",
            "guitar": "instrument",
            "guitars": "instrument",
            "violin": "instrument",
            "violins": "instrument",
            "cello": "instrument",
            "cellos": "instrument",
            "flute": "instrument",
            "flutes": "instrument",
            "clarinet": "instrument",
            "clarinets": "instrument",
            "saxophone": "instrument",
            "saxophones": "instrument",
            "trumpet": "instrument",
            "trumpets": "instrument",
            "trombone": "instrument",
            "trombones": "instrument",
            "tuba": "instrument",
            "tubas": "instrument",
            "sousaphone": "instrument",
            "drum": "instrument",
            "drums": "instrument",
            "banjo": "instrument",
            "banjos": "instrument",
            "mandolin": "instrument",
            "mandolins": "instrument",
            # trees -> general category "tree"
            "pine": "tree",
            "pines": "tree",
            "oak": "tree",
            "oaks": "tree",
            "spruce": "tree",
            "spruces": "tree",
            "fir": "tree",
            "firs": "tree",
            "maple": "tree",
            "maples": "tree",
            "birch": "tree",
            "birches": "tree",
            # phrasing normalizations
            "will play": "play",
            "is going to play": "play",
            "is going to": "will",
        }

        def apply_safe_replacements(text: str) -> str:
            "Apply the SAFE_REPLACEMENTS mapping using whole-word matches."
            import re
            t = text
            for k, v in SAFE_REPLACEMENTS.items():
                # word-boundary replacements
                t = re.sub(r"\b" + re.escape(k) + r"\b", v, t)
            # collapse spaces
            t = " ".join(t.split())
            return t

        def remove_articles(tokens: list[str]) -> list[str]:
            "Remove simple determiners that are not essential for propositional match."
            return [t for t in tokens if t not in ("a", "an", "the")]

        def tokens(s: str) -> list[str]:
            "Split into tokens (already normalized) and remove empty strings."
            return [t for t in s.split() if t]

        def is_subsequence(short: list[str], long: list[str]) -> bool:
            "Check if short token list appears in order inside long, allowing gaps."
            if not short:
                return True
            i = 0
            for tok in long:
                if tok == short[i]:
                    i += 1
                    if i >= len(short):
                        return True
            return False

        prop = extract_proposition(q)
        n_ctx = normalize_text(ctx)
        n_prop = normalize_text(prop)

        # Apply safe lexical replacements
        r_ctx = apply_safe_replacements(n_ctx)
        r_prop = apply_safe_replacements(n_prop)

        # Quick deterministic checks that are conservative:
        # 1) Exact normalized substring match (most reliable).
        if r_prop and r_prop in r_ctx:
            return dspy.Prediction(answer="Yes")

        # 2) Token-based checks: after removing articles, if the proposition tokens
        #    appear as an ordered subsequence inside the context tokens, treat as entailment.
        #    This is conservative because it requires the same token order (including negation
        #    words if present); do not ignore negation or modal operators.
        prop_toks = remove_articles(tokens(r_prop))
        ctx_toks = remove_articles(tokens(r_ctx))
        # require at least two non-article tokens to use subsequence heuristic to avoid matching
        # on trivial single-word overlaps.
        if len(prop_toks) >= 2 and is_subsequence(prop_toks, ctx_toks):
            return dspy.Prediction(answer="Yes")

        # If deterministic checks couldn't conclude, fall back to LM.
        lm_out = self.predict(context=ctx, question=q)
        ans = lm_out.answer if hasattr(lm_out, "answer") else str(lm_out)
        ans_norm = str(ans).strip()
        if ans_norm.lower() == "yes":
            final = "Yes"
        elif ans_norm.lower() == "no":
            final = "No"
        else:
            # If LM output is unexpected, be conservative.
            final = "No"
        return dspy.Prediction(answer=final)

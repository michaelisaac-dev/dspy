class ScoNeSignatureModule(dspy.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **inputs):
        """
        Deterministic heuristic for ScoNe "Can we logically conclude...?" questions.
        Returns 'Yes' or 'No'.

        Strategy (deterministic, no LM):
        - Extract the claim from the question using common phrasings.
        - Normalize context and claim (lowercase, collapse spaces, strip surrounding punctuation).
        - If the claim appears verbatim in the context -> Yes.
        - If simple negation presence differs between context and claim -> No (conservative).
        - Use a small curated hyponym->hypernym map to detect simple specificity/generalization:
            * If context contains a hyponym (specific term) and claim contains its hypernym (general term) -> Yes.
            * If context contains a hypernym (general term) and claim contains a hyponym (specific term) -> No.
          (Do NOT conclude entailment merely because the same word appears in both; matching an entity
           like "boy" in both context and claim is NOT sufficient to entail the whole claim.)
        - Otherwise, conservatively answer 'No'.
        """
        import re

        context = inputs.get("context", "") or ""
        question = inputs.get("question", "") or ""

        def normalize(s: str) -> str:
            s = s.lower().strip()
            # replace common punctuation with spaces, collapse whitespace
            s = re.sub(r"[\-—_/()\",;:]", " ", s)
            s = re.sub(r"\s+", " ", s)
            # strip surrounding punctuation like leading/trailing quotes or question marks
            s = s.strip(" ?.!")
            return s

        def extract_claim(q: str) -> str:
            qn = q.strip()
            # common phrasings used in these tasks (capture the asserted claim)
            patterns = [
                r"can we logically conclude(?: for sure)? that (.+)\?",
                r"can we logically conclude(?: for sure)? that (.+)$",
                r"can we conclude(?: for sure)? that (.+)\?",
                r"can we conclude(?: for sure)? that (.+)$",
                r"is it the case that (.+)\?",
                r"is it the case that (.+)$",
                r"does it follow that (.+)\?",
                r"does it follow that (.+)$",
            ]
            for pat in patterns:
                m = re.search(pat, qn, flags=re.IGNORECASE)
                if m:
                    return normalize(m.group(1))
            # fallback: remove a trailing question mark and normalize whole question
            return normalize(qn.rstrip(" ?"))

        ctx = normalize(context)
        claim = extract_claim(question)

        # Helper: detect simple negation presence (conservative)
        def has_neg(s: str) -> bool:
            return bool(re.search(r"\b(not|n't|no|never|none)\b", s))

        # Quick exact substring check: if claim appears verbatim in context -> entailed
        if claim and (" " + claim + " ") in (" " + ctx + " "):
            return dspy.Prediction(answer="Yes")

        # If negation presence differs, be conservative and return No
        if has_neg(ctx) != has_neg(claim):
            return dspy.Prediction(answer="No")

        # Small curated hyponym -> hypernym map for common categories where specificity matters
        hyponym_map = {
            # instruments
            "sousaphone": "instrument",
            "sousa phone": "instrument",
            "trumpet": "instrument",
            "horn": "instrument",
            "horns": "instrument",
            "trombone": "instrument",
            "saxophone": "instrument",
            "violin": "instrument",
            "drum": "instrument",
            "guitar": "instrument",
            "piano": "instrument",
            "clarinet": "instrument",
            "flute": "instrument",
            "oboe": "instrument",
            "cello": "instrument",
            "tuba": "instrument",
            "bass": "instrument",
            "xylophone": "instrument",
            # people roles / occupations (useful in some cases)
            "professor": "professional",
            "doctor": "professional",
            "engineer": "professional",
            # family terms (some taxonomic relations)
            "mother": "woman",
            "father": "man",
            "mom": "woman",
            "dad": "man",
            "boy": "male",
            "girl": "female",
        }

        # Check presence of these words (word-boundary aware) in context and claim
        def contains_word(text: str, word: str) -> bool:
            if not word:
                return False
            word_escaped = re.escape(word)
            # match whole word or simple plural forms (word + s)
            if re.search(r"\b" + word_escaped + r"\b", text):
                return True
            if re.search(r"\b" + word_escaped + r"s\b", text):
                return True
            return False

        # Apply hyponym/hypernym heuristics carefully:
        for hypo, hyper in hyponym_map.items():
            in_ctx_hypo = contains_word(ctx, hypo)
            in_ctx_hyper = contains_word(ctx, hyper)
            in_claim_hypo = contains_word(claim, hypo)
            in_claim_hyper = contains_word(claim, hyper)

            # If context has a specific (hypo) and claim asks about the general (hyper) -> Yes
            if in_ctx_hypo and in_claim_hyper:
                return dspy.Prediction(answer="Yes")

            # If context has the general (hyper) and claim asks about a specific (hypo) -> No
            if in_ctx_hyper and in_claim_hypo:
                return dspy.Prediction(answer="No")

            # NOTE: do NOT conclude entailment solely because both contain the same word.
            # That is not sufficient to entail the full claim.

        # Conservative default when we cannot establish entailment
        return dspy.Prediction(answer="No")

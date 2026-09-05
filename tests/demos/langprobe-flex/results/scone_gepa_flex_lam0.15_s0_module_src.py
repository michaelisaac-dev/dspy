class ScoNeSignatureModule(dspy.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **inputs):
        """
        Deterministic, conservative entailment heuristics (no LM).
        Return "Yes" only when the question's core claim is clearly entailed
        by the context according to simple lexical/subclass rules or exact match.
        Otherwise return "No".

        Heuristics used:
        - If the normalized question statement is an exact substring of the normalized context -> Yes.
        - If the context contains a specific hyponym (e.g. "nerd", "pastry") and the question
          asks about its hypernym (e.g. "person", "food") -> Yes.
        - If the question contains a hyponym (more specific term, e.g. "folk") but the context
          only mentions the hypernym (e.g. "music") -> No (question is more specific than stated).
        - If most (>=80%) of the content words in the statement appear wordwise in the context -> Yes.
          This is a conservative fallback and is only used if the above lexical/hierarchy tests don't decide.
        - Otherwise return No (conservative).
        """
        import re

        context = inputs.get("context", "") or ""
        question = inputs.get("question", "") or ""

        def normalize(text: str) -> str:
            # Lowercase, remove punctuation (keep word boundaries and apostrophes inside words),
            # collapse whitespace.
            t = text.lower()
            t = re.sub(r"[^\w\s']", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
            return t

        def extract_statement(q: str) -> str:
            # Remove common interrogative framing so the core statement remains.
            q = q.strip()
            q_low = q.lower()
            patterns = [
                r"^can we (?:logically )?conclude(?: for sure)? that (.*)\??$",
                r"^can we conclude that (.*)\??$",
                r"^is it correct that (.*)\??$",
                r"^is it true that (.*)\??$",
                r"^does the context imply that (.*)\??$",
                r"^do we know that (.*)\??$",
                r"^(.*)\?$",  # fallback: strip trailing ?
            ]
            for pat in patterns:
                m = re.match(pat, q_low, flags=re.IGNORECASE)
                if m:
                    return m.group(1).strip()
            return q_low

        # Expand the small deterministic hyponym -> hypernym map to cover common cases
        # (genres, roles, food items, animal young->adult, etc.). This lets the module
        # detect when the question asks for a more specific thing than the context states.
        hyponym_to_hypernym = {
            # people / roles
            "nerd": "person",
            "man": "person",
            "woman": "person",
            "boy": "person",
            "girl": "person",
            "child": "person",
            "kid": "person",
            "adult": "person",
            "teen": "person",
            "musher": "person",
            # food/items
            "pastry": "food",
            "cake": "food",
            "cookie": "food",
            "bread": "food",
            "croissant": "food",
            "pizza": "food",
            "burrito": "food",
            # music genres -> music
            "folk": "music",
            "classical": "music",
            "jazz": "music",
            "rock": "music",
            "pop": "music",
            "hiphop": "music",
            "hip-hop": "music",
            "rap": "music",
            "country": "music",
            "blues": "music",
            "metal": "music",
            "electronic": "music",
            "edm": "music",
            "punk": "music",
            "reggae": "music",
            # animals / pets
            "puppy": "dog",
            "pup": "dog",
            "kitten": "cat",
            # transport / vehicles
            "sedan": "car",
            "hatchback": "car",
            # generic small mapping
            "dog": "animal",
            "cat": "animal",
        }

        def contains_word(text: str, word: str) -> bool:
            # match the exact whole word (word boundary)
            return re.search(r"\b" + re.escape(word) + r"\b", text) is not None

        norm_context = normalize(context)
        statement = extract_statement(question)
        norm_statement = normalize(statement)

        # 0) Quick guard: if either is empty, be conservative
        if not norm_statement or not norm_context:
            return dspy.Prediction(answer="No")

        # 1) Exact substring entailment (fast check)
        if norm_statement in norm_context:
            return dspy.Prediction(answer="Yes")

        # 2) Hyponym/hypernym heuristics
        # 2a) If the context contains a hyponym and the statement asks about its hypernym -> Yes
        for hyponym, hyper in hyponym_to_hypernym.items():
            if contains_word(norm_context, hyponym) and contains_word(norm_statement, hyper):
                return dspy.Prediction(answer="Yes")

        # 2b) If the statement contains a hyponym but context only mentions the hypernym -> No
        #     (the question is asking for a more specific claim than the context supports).
        for hyponym, hyper in hyponym_to_hypernym.items():
            if contains_word(norm_statement, hyponym) and contains_word(norm_context, hyper):
                # If the context also contains the hyponym word itself, we've already handled that above.
                # Here the context only has the general term, so the specific claim is not entailed.
                if not contains_word(norm_context, hyponym):
                    return dspy.Prediction(answer="No")

        # 3) Content-word overlap fallback: require most content words to appear wordwise.
        stmt_words = [w for w in norm_statement.split() if len(w) > 2]
        if stmt_words:
            matched = sum(1 for w in stmt_words if contains_word(norm_context, w))
            # require most words to match (>= 80%)
            if (matched / len(stmt_words)) >= 0.8:
                return dspy.Prediction(answer="Yes")

        # Default conservative answer
        return dspy.Prediction(answer="No")

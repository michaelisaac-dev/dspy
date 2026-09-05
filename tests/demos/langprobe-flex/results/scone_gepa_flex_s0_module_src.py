class ScoNeSignatureModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use a ChainOfThought predictor so the model explicitly reasons about logical entailment.
        # The predictor must output a final, single token answer: "Yes" or "No".
        self.classify = dspy.ChainOfThought(dspy.Signature(
            "context: str, question: str -> answer: str",
            (
                "Task: Decide whether the QUESTION is logically entailed by the CONTEXT. "
                "Answer exactly 'Yes' if the question's statement is true in every possible situation "
                "consistent with the context (i.e., logically entailed). Otherwise answer exactly 'No'.\n\n"
                "Procedure (required):\n"
                "1) Paraphrase the CONTEXT as a precise logical constraint (identify universal/ existential "
                "claims and the scope of any negation: e.g. 'there is no way that X' or 'it is impossible that X' "
                "means X is false in all possible situations).\n"
                "2) Paraphrase the QUESTION as a proposition to test against that constraint.\n"
                "3) Decide whether every model satisfying the CONTEXT must also satisfy the QUESTION. If yes, "
                "the correct answer is 'Yes'. If there exists any situation consistent with the CONTEXT where the "
                "QUESTION is false, the correct answer is 'No'.\n\n"
                "Lexical guidance: you may use common, well-known class membership (hyponymy) such as "
                "'a willow is a kind of tree' to propagate entailment (e.g., 'is next to a willow' -> 'is next to a tree'). "
                "Do NOT invent obscure lexical facts; when a lexical relation is ambiguous or unclear, be conservative and answer 'No'.\n\n"
                "Negation rules to follow explicitly:\n"
                "- From 'must be next to a willow' you may conclude 'must be next to a tree' (hyponym -> hypernym).\n"
                "- From 'must not be next to a willow' you CANNOT conclude 'must not be next to a tree' (not next to one subtype "
                "does not imply not next to all supertypes).\n\n"
                "Output format requirement: produce a short chain-of-thought showing your reasoning steps, "
                "and ensure the final line contains ONLY the single word 'Yes' or 'No' (capitalized, no punctuation). "
                "The predictor's 'answer' field must contain that exact final token."
            )
        ))

    def forward(self, **inputs):
        result = self.classify(context=inputs["context"], question=inputs["question"])
        # Coerce and validate the answer: accept only 'Yes' or 'No' (case-insensitive).
        raw = (result.answer or "").strip()
        # Normalize common patterns: sometimes model appends 'Answer: Yes' or similar.
        import re
        m = re.findall(r"\b(yes|no)\b", raw, flags=re.IGNORECASE)
        if m:
            final = m[-1].lower().capitalize()
        else:
            # If the model failed to include an explicit Yes/No token, be conservative and answer 'No'.
            final = "No"
        return dspy.Prediction(answer=final)

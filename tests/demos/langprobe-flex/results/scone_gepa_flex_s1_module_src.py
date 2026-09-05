class ScoNeSignatureModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use ChainOfThought so the model can show its reasoning steps for harder scope/quantifier cases.
        # The instructions below were refined from failing examples to (a) explicitly explain
        # how to treat quantifiers and double negation, (b) state the correct hyponym/hypernym rules
        # for existence and positive assertions, and (c) correct the earlier mistaken guidance about
        # negation+hyponymy.
        self.entail = dspy.ChainOfThought(dspy.Signature(
            "context: str, question: str -> answer: str",
            (
                "Task: Decide whether the QUESTION is strictly logically ENTAILED by the CONTEXT. "
                "Return exactly one token, either 'Yes' or 'No' (capitalized), and nothing else.\n\n"
                "Definition of entailment: 'Entailed' means that in every possible situation (world) "
                "where the CONTEXT is true, the QUESTION must also be true. Answer 'Yes' only when the "
                "QUESTION follows in all such worlds; otherwise answer 'No'. Be conservative.\n\n"
                "Precise rules the model must follow when judging entailment:\n"
                "1) Hyponym/hypernym (subtype) rule for positive assertions and existence:\n"
                "   - If the CONTEXT asserts that an individual or event involves a specific subtype A, "
                "     and the QUESTION asserts the corresponding supertype B (A is a kind of B), then the "
                "     CONTEXT entails the QUESTION. Example pattern: 'fell off a racehorse' -> 'fell off a horse'.\n"
                "   - Conversely, a general term in the CONTEXT (e.g. 'tree') does NOT entail a specific subtype "
                "     in the QUESTION (e.g. 'maple').\n\n"
                "2) Quantifiers and negation handling (explicit rules):\n"
                "   - Phrases like 'has seen any X', 'there is an X', 'there is at least one X' indicate existence (∃X).\n"
                "   - Phrases like 'has not seen any X' or 'there is no X' indicate non-existence (¬∃X).\n"
                "   - A surrounding 'It is not the case that (has not ...)' is a double negation that yields existence. "
                "     For example, 'It is not the case that the diver has not seen any fish' entails 'the diver has "
                "     seen some fish' (∃ fish).\n"
                "   - Existence of a subtype implies existence of a supertype: if CONTEXT implies ∃A and A is a subtype of B, "
                "     then CONTEXT entails ∃B. Use common, widely-known subtype relations (e.g. fish -> creature, racehorse -> horse).\n\n"
                "3) Negation + hyponymy caution: do NOT assume that negating a subtype entails negating a supertype. "
                "   For example, 'not next to a cedar' does NOT entail 'not next to a tree' (there could be other trees).\n\n"
                "4) No invention of facts: do not assume anything not stated. If the QUESTION could be false in some world "
                "   compatible with the CONTEXT, answer 'No'. Ignore pragmatic implicatures or conversational defaults.\n\n"
                "5) Output format: EXACTLY one token, either 'Yes' or 'No' (capitalized), and nothing else. "
                "   Do not add explanation in the final token. Use the chain-of-thought channel only for internal reasoning; "
                "   the 'answer' field must be strictly 'Yes' or 'No'."
            )
        ))

    def forward(self, **inputs):
        # Call the LM predictor with the provided fields
        res = self.entail(context=inputs["context"], question=inputs["question"])
        # Normalize and map the LM's raw answer to exactly 'Yes' or 'No'.
        raw = (res.answer or "").strip()
        low = raw.lower()

        yes_set = {"yes", "y", "entailed", "entailed.", "true"}
        no_set = {"no", "n", "not", "not_entailed", "not entailed", "contradicted", "unknown", "undetermined"}

        # Robust checks: startswith 'y' or 'n' handles single-letter replies like 'Y'/'N'.
        if low.startswith("y") or low in yes_set:
            answer = "Yes"
        elif low.startswith("n") or low in no_set:
            answer = "No"
        else:
            # Be conservative on any unclear or unexpected output: default to 'No'
            answer = "No"

        return dspy.Prediction(answer=answer)

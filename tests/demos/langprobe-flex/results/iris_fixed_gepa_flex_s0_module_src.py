class SigModule(dspy.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **inputs):
        """
        Deterministic iris classifier that uses numeric rules (no LLM).
        Primary signal: petal_length (most discriminative).
        Fallbacks: petal_width, then a crude sepal-based heuristic, then a safe default.

        Returns one of the exact strings: 'setosa', 'versicolor', 'virginica'.
        """
        def to_float(value):
            """Parse a numeric string like '5.6' or '5,6' -> float. Return None if not parseable."""
            if value is None:
                return None
            s = str(value).strip()
            if s == "":
                return None
            # Accept comma as decimal separator
            s = s.replace(",", ".")
            try:
                return float(s)
            except Exception:
                return None

        pl = to_float(inputs.get("petal_length"))
        pw = to_float(inputs.get("petal_width"))
        sl = to_float(inputs.get("sepal_length"))
        sw = to_float(inputs.get("sepal_width"))

        # Primary deterministic rule (petal length thresholds are standard and highly discriminative):
        if pl is not None:
            if pl <= 2.5:
                return dspy.Prediction(answer="setosa")
            elif pl <= 4.8:
                return dspy.Prediction(answer="versicolor")
            else:
                return dspy.Prediction(answer="virginica")

        # Fallback 1: petal width thresholds
        if pw is not None:
            if pw <= 0.6:
                return dspy.Prediction(answer="setosa")
            elif pw <= 1.8:
                return dspy.Prediction(answer="versicolor")
            else:
                return dspy.Prediction(answer="virginica")

        # Fallback 2: simple sepal-based heuristic if both sepal measurements available
        if sl is not None and sw is not None:
            # Heuristic: small sepal length + fairly large sepal width often indicates setosa.
            if sl < 5.0 and sw > 3.0:
                return dspy.Prediction(answer="setosa")
            elif sl < 6.5:
                return dspy.Prediction(answer="versicolor")
            else:
                return dspy.Prediction(answer="virginica")

        # Last resort: choose versicolor as a safe default when inputs are missing/unparseable.
        return dspy.Prediction(answer="versicolor")

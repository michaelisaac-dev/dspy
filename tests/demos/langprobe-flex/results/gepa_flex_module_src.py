class HeartDiseaseSignatureModule(dspy.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **inputs):
        """
        Deterministic, conservative heuristic classifier for presence of heart disease.
        Returns answer: 'yes' or 'no'.

        Revision notes (from failing examples):
         - The previous weights produced too many positives. This version is more conservative:
           reduce the influence of exercise-induced angina (exang) and number of vessels (ca),
           require a higher combined score to predict 'yes'.
         - We parse robustly from strings, accept common truthy/falsey forms, and coerce numbers.
         - Decision rule produces only 'yes' or 'no' (lowercase).
        """

        def _norm_str(x: object) -> str:
            if x is None:
                return ""
            return str(x).strip().lower()

        def _to_float(x: object, default: float = 0.0) -> float:
            try:
                return float(str(x).strip())
            except Exception:
                return default

        def _to_int(x: object, default: int = 0) -> int:
            try:
                return int(float(str(x).strip()))
            except Exception:
                return default

        def _is_true(x: object) -> bool:
            s = _norm_str(x)
            return s in ("1", "true", "t", "yes", "y")

        # Normalize inputs
        exang = _is_true(inputs.get("exang", ""))
        oldpeak = _to_float(inputs.get("oldpeak", 0.0))
        ca = _to_int(inputs.get("ca", 0))
        thalach = _to_float(inputs.get("thalach", 0.0))
        age = _to_int(inputs.get("age", 0))
        chol = _to_float(inputs.get("chol", 0.0))
        fbs = _is_true(inputs.get("fbs", "0"))
        restecg = _norm_str(inputs.get("restecg", ""))
        thal = _norm_str(inputs.get("thal", ""))

        # Conservative weights chosen to reduce false positives observed in feedback.
        # These are interpretable contributions (floats); higher total increases likelihood of disease.
        score = 0.0

        # Exercise-induced angina (reduced weight compared to earlier version)
        if exang:
            score += 1.0

        # ST depression: stronger if >= 2.0 (more worrisome)
        if oldpeak >= 2.0:
            score += 1.5
        elif oldpeak >= 1.0:
            score += 0.5

        # Number of major vessels: moderate weight (ca in [1,3])
        if ca >= 2:
            score += 1.5
        elif ca == 1:
            score += 0.8

        # Max heart rate achieved: low max HR is a risk indicator
        if thalach < 120:
            score += 1.0
        elif thalach < 130:
            score += 0.5

        # Age: older age contributes slightly
        if age >= 75:
            score += 1.0
        elif age >= 65:
            score += 0.5

        # Cholesterol: elevated cholesterol adds modestly
        if chol >= 280:
            score += 1.0
        elif chol >= 240:
            score += 0.5

        # Fasting blood sugar
        if fbs:
            score += 0.5

        # ECG / thal minor signals
        if "abnormal" in restecg or "hypertrophy" in restecg or "st-t" in restecg:
            score += 0.5
        if thal in ("fixed", "reversible"):
            score += 0.5
        # Note: 'normal' thal contributes nothing.

        # Decision threshold — set conservatively to require stronger combined evidence
        threshold = 3.0
        answer = "yes" if score >= threshold else "no"

        return dspy.Prediction(answer=answer)

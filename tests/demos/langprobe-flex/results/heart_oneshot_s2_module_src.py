class HeartDiseaseSignatureModule(dspy.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **inputs):
        # Helper: safe parsing and normalization
        def norm_str(key: str) -> str:
            v = inputs.get(key, "")
            if v is None:
                return ""
            return str(v).strip().lower()

        def to_float(x, default=0.0):
            try:
                return float(str(x).strip())
            except Exception:
                return default

        def to_int(x, default=0):
            try:
                return int(float(str(x).strip()))
            except Exception:
                return default

        # Read and normalize fields
        age = to_int(inputs.get("age", ""), 0)
        sex = norm_str("sex")
        cp = norm_str("cp")
        trestbps = to_float(inputs.get("trestbps", ""), 0.0)
        chol = to_float(inputs.get("chol", ""), 0.0)
        fbs_raw = norm_str("fbs")
        # Accept "1"/"0", "true"/"false"
        fbs = 1 if fbs_raw in ("1", "true", "yes", "y", "t") else 0
        restecg = norm_str("restecg")
        thalach = to_float(inputs.get("thalach", ""), 0.0)
        exang = norm_str("exang")
        oldpeak = to_float(inputs.get("oldpeak", ""), 0.0)
        slope = norm_str("slope")
        ca_raw = norm_str("ca")
        # ca may be numeric or textual; try int parse
        ca = to_int(ca_raw, 0)
        thal = norm_str("thal")

        # Scoring heuristic built from observed patterns in examples:
        # Each feature adds/subtracts points; final threshold determines presence.
        score = 0.0

        # Exercise-induced angina contributes moderate risk but is not decisive alone.
        if exang in ("yes", "y", "true"):
            score += 1.2

        # ST depression (oldpeak) is an important continuous risk signal.
        score += float(oldpeak)  # 1 point per unit of oldpeak

        # Slope: upsloping is low risk, flat moderate, downsloping higher.
        if "flat" in slope:
            score += 0.6
        elif "down" in slope:
            score += 1.2

        # Number of major vessels seen: more vessels -> more risk, moderate weight.
        if ca >= 0:
            score += ca * 0.6

        # Max heart rate achieved (thalach): lower max HR often indicates risk.
        if thalach > 0:
            if thalach < 120:
                score += 1.6
            elif thalach <= 140:
                score += 0.6

        # Resting ECG abnormalities add modest risk.
        if "st-t" in restecg or "st-t wave" in restecg or "st-t wave abnormality" in restecg or "st-t" in restecg:
            score += 0.7
        if "left ventricular" in restecg or "left ventricular hypertrophy" in restecg or "lvh" in restecg:
            score += 0.7

        # Chest pain: asymptomatic chest pain tends to appear in higher-risk examples.
        if "asymptomatic" in cp:
            score += 0.4

        # Thalassemia coded values: reversible shows higher risk in examples.
        if "reversible" in thal or "revers" in thal:
            score += 0.7
        elif "fixed" in thal:
            score += 0.2

        # Older age adds small weight.
        if age >= 60:
            score += 0.5

        # High resting blood pressure and high cholesterol contribute small incremental risk.
        if trestbps >= 160:
            score += 0.6
        elif trestbps >= 140:
            score += 0.3

        if chol >= 300:
            score += 0.5
        elif chol >= 240:
            score += 0.2

        # Fasting blood sugar > 120 (fbs == 1) small contribution
        if fbs == 1:
            score += 0.3

        # Final decision threshold chosen to balance sensitivity vs specificity seen in examples.
        answer = "yes" if score >= 5.0 else "no"

        return dspy.Prediction(answer=answer)

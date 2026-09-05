class HeartDiseaseModule(dspy.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **inputs):
        # Helpers for robust parsing
        def norm(s):
            return "" if s is None else str(s).strip().lower()

        def parse_float(s, default=None):
            s = norm(s)
            if s == "":
                return default
            try:
                return float(s)
            except Exception:
                # try to extract numeric portion
                import re
                m = re.search(r"[-+]?\d*\.?\d+", s)
                if m:
                    try:
                        return float(m.group(0))
                    except Exception:
                        return default
                return default

        def parse_int(s, default=None):
            s = norm(s)
            if s == "":
                return default
            try:
                return int(float(s))
            except Exception:
                import re
                m = re.search(r"[-+]?\d+", s)
                if m:
                    try:
                        return int(m.group(0))
                    except Exception:
                        return default
                return default

        def is_yes(s):
            s = norm(s)
            return s in ("yes", "y", "true", "1", "t")

        # Normalize inputs
        age = parse_int(inputs.get("age"))
        sex = norm(inputs.get("sex"))
        cp = norm(inputs.get("cp"))
        trestbps = parse_int(inputs.get("trestbps"))
        chol = parse_int(inputs.get("chol"))
        fbs = norm(inputs.get("fbs"))
        restecg = norm(inputs.get("restecg"))
        thalach = parse_int(inputs.get("thalach"))
        exang = norm(inputs.get("exang"))
        oldpeak = parse_float(inputs.get("oldpeak"), default=0.0)
        slope = norm(inputs.get("slope"))
        ca_raw = norm(inputs.get("ca"))
        thal_raw = norm(inputs.get("thal"))

        # Interpret ca (number of vessels). Many inputs are "0","1","2","3" or textual.
        ca = None
        if ca_raw != "":
            try:
                ca = int(float(ca_raw))
            except Exception:
                # sometimes non-numeric; try to extract a digit
                import re
                m = re.search(r"\d+", ca_raw)
                if m:
                    ca = int(m.group(0))
        if ca is None:
            ca = 0

        # Detect thal abnormality by textual cues (prefer explicit words like 'revers' or 'fixed' or 'defect')
        thal = thal_raw
        thal_abnormal = False
        if thal:
            if any(k in thal for k in ("revers", "fixed", "defect")):
                thal_abnormal = True

        # Build a simple, transparent clinical risk score.
        # We designed weights to reflect features that increase probability of heart disease.
        score = 0

        # Exercise-induced angina (yes) increases likelihood
        if is_yes(exang):
            score += 2

        # ST depression (oldpeak) large values indicate ischemia
        # threshold chosen to separate mild vs significant depression
        if oldpeak is not None and oldpeak >= 2.5:
            score += 3

        # Number of vessels colored: 3 is a strong signal; 2 a mild signal
        if ca >= 3:
            score += 3
        elif ca == 2:
            score += 1

        # Abnormal thalassemia findings (reversible/fixed defect) add modest weight
        if thal_abnormal:
            score += 1

        # Unfavorable slope adds small weight
        if slope in ("flat", "downsloping", "downslop", "down"):
            score += 1

        # Asymptomatic chest pain (lack of typical angina) is sometimes associated with silent ischemia in datasets
        if "asymptom" in cp:  # catch 'asymptomatic'
            score += 1

        # Low maximum heart rate achieved (inability to reach high HR) adds some weight
        if thalach is not None and thalach < 120:
            score += 1

        # Very high cholesterol adds small weight
        if chol is not None and chol >= 300:
            score += 1

        # Final decision threshold chosen to balance sensitivity for clear signals
        answer = "yes" if score >= 4 else "no"
        return dspy.Prediction(answer=answer)

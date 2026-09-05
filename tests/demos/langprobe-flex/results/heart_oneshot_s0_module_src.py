class HeartDiseaseModule(dspy.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **inputs):
        # Deterministic, rule-based risk scoring derived from common clinical risk features.
        # Returns "yes" if estimated risk score is high, otherwise "no".
        def norm(s: str) -> str:
            return (s or "").strip().lower()

        def to_float(s: str, default: float = 0.0) -> float:
            try:
                return float(s)
            except Exception:
                # some fields might be like "0" or "1" or words; fallback 0
                return default

        def to_int(s: str, default: int = 0) -> int:
            try:
                return int(float(s))
            except Exception:
                return default

        def yesno_to_bool(s: str) -> bool:
            s2 = norm(s)
            if s2 in {"1", "true", "t", "yes", "y"}:
                return True
            if s2 in {"0", "false", "f", "no", "n"}:
                return False
            # unknown -> False
            return False

        age = to_float(inputs.get("age", "0"))
        sex = norm(inputs.get("sex", ""))
        cp = norm(inputs.get("cp", ""))
        trestbps = to_float(inputs.get("trestbps", "0"))
        chol = to_float(inputs.get("chol", "0"))
        fbs_raw = inputs.get("fbs", "")
        fbs = yesno_to_bool(fbs_raw)
        restecg = norm(inputs.get("restecg", ""))
        thalach = to_float(inputs.get("thalach", "0"))
        exang = yesno_to_bool(inputs.get("exang", ""))
        oldpeak = to_float(inputs.get("oldpeak", "0"))
        slope = norm(inputs.get("slope", ""))
        ca = to_int(inputs.get("ca", "0"))
        thal = norm(inputs.get("thal", ""))

        # Normalize some variant tokens that appear in datasets
        # cp may be numeric codes in some inputs — treat unknown codes as neutral
        if cp in {"0", "1", "2", "3"}:
            # common coding: 0-3 map to angina types; treat "0" or "3" as less specific
            # fall back to descriptive mapping if possible; otherwise keep as-is
            # do not assume mapping; keep neutral for unknown numeric codes
            cp_cat = cp
        else:
            cp_cat = cp

        # Map restecg synonyms
        if "st-t" in restecg or ("st" in restecg and "t" in restecg):
            restecg_cat = "st-t"
        elif "ventricular" in restecg or "hypertrophy" in restecg or "lvh" in restecg:
            restecg_cat = "lvh"
        else:
            restecg_cat = "normal" if restecg.strip() == "normal" else restecg

        # Map thal synonyms
        if "normal" in thal:
            thal_cat = "normal"
        elif "fixed" in thal:
            thal_cat = "fixed"
        elif "revers" in thal:
            thal_cat = "reversible"
        elif thal in {"1", "2", "3"}:
            # numeric thal codes sometimes appear; treat "3" or "2" as abnormal conservatively
            thal_cat = "abnormal"
        else:
            thal_cat = thal

        # Start scoring
        score = 0.0

        # Age: modest weight
        if age > 60:
            score += 0.5
        elif age >= 50:
            score += 0.25

        # Resting blood pressure
        if trestbps > 140:
            score += 0.5

        # Cholesterol
        if chol > 240:
            score += 0.5

        # Fasting blood sugar
        if fbs:
            score += 0.25

        # Resting ECG abnormalities
        if restecg_cat in {"st-t", "lvh"}:
            score += 0.5

        # Max heart rate achieved: low values are worse
        if thalach < 100:
            score += 2.0
        elif thalach < 120:
            score += 1.0
        elif thalach < 140:
            score += 0.5
        # >140 -> 0

        # Exercise-induced angina
        if exang:
            score += 1.0

        # ST depression (oldpeak)
        if oldpeak > 3.5:
            score += 1.75
        elif oldpeak > 2.5:
            score += 1.25
        elif oldpeak > 1.5:
            score += 0.75
        elif oldpeak > 0.5:
            score += 0.25

        # Slope of peak exercise ST segment
        if "down" in slope:
            score += 0.5
        elif "flat" in slope:
            score += 0.25
        # upsloping -> 0

        # Number of major vessels colored by fluoroscopy
        if ca > 0:
            # each vessel modestly increases score
            score += 0.5 * float(min(ca, 3))

        # Thalassemia / perfusion defects
        if thal_cat in {"fixed", "reversible", "abnormal"}:
            score += 0.5

        # Chest pain: certain types slightly adjust risk (typical angina often more concerning)
        if "typical" in cp_cat:
            score += 0.5
        elif "asymptomatic" in cp_cat:
            # asymptomatic chest pain may be neutral; no change
            score += 0.0
        elif "non-anginal" in cp_cat:
            score += 0.0
        elif "atypical" in cp_cat:
            score += 0.25
        # numeric/unknown -> no addition

        # Decision threshold: require a score strictly greater than 4.0 to call "yes".
        # This conservative threshold avoids false positives on borderline cases.
        answer = "yes" if score > 4.0 else "no"

        return dspy.Prediction(answer=answer)

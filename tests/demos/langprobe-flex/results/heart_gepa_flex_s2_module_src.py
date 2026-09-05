class HeartDiseaseSignatureModule(dspy.Module):
    """
    Given patient information (features as strings), predict presence of heart disease
    using a conservative, deterministic rule-based decision.

    Decision summary (conservative, multi-factor rule):
    - We compute several independent abnormal indicators (vessel_flag, typical chest pain,
      exercise-ischemia pattern, abnormal resting ECG). Each counts as one indicator.
    - Abnormal thal (fixed/reversible defect) only increases suspicion when at least one
      vessel is flagged (ca >= 1) — it does not by itself count as an indicator.
    - Large ST depression (oldpeak > 3.0) is counted as an indicator, but we require
      at least two total indicators before returning "yes" so that oldpeak alone
      does not force a positive.
    - Final rule: return "yes" only when the total indicator count is >= 2.
      This avoids single-factor false positives (e.g., a lone ca=2 with no other
      abnormalities will not force "yes").
    - The module is robust to common numeric/boolean string encodings and accepts
      either descriptive strings or numeric codes where present.
    - Output is exactly "yes" or "no".
    """
    def __init__(self):
        super().__init__()

    def forward(self, **inputs):
        # -- Helpers --
        def as_int(x, default=None):
            if x is None:
                return default
            try:
                return int(float(str(x).strip()))
            except Exception:
                return default

        def as_float(x, default=None):
            if x is None:
                return default
            try:
                return float(str(x).strip())
            except Exception:
                return default

        def norm(s):
            if s is None:
                return ""
            return str(s).strip().lower()

        def parse_bool_like(s):
            """Return True/False/None for common encodings."""
            if s is None:
                return None
            t = norm(s)
            if t in ("1", "true", "t", "yes", "y"):
                return True
            if t in ("0", "false", "f", "no", "n"):
                return False
            return None

        def contains_token(field, tokens):
            """Case-insensitive substring check for any token (tokens may be str or iterable)."""
            f = norm(field)
            if isinstance(tokens, str):
                tokens = (tokens,)
            for tok in tokens:
                if not tok:
                    continue
                if tok in f:
                    return True
            return False

        # Map common numeric chest pain encodings to descriptive tokens when present.
        # Many datasets encode cp as numbers (1=typical, 2=atypical, 3=non-anginal, 4=asymptomatic).
        def cp_is_typical(cp_field):
            t = norm(cp_field)
            if not t:
                return False
            if contains_token(t, ("typical", "typical angina")):
                return True
            # handle common numeric encodings if present
            if t in ("1", "01"):  # common encoding where 1 => typical angina
                return True
            return False

        # -- Parse and normalize inputs --
        age = as_int(inputs.get("age"))
        sex = norm(inputs.get("sex"))
        cp = norm(inputs.get("cp"))
        trestbps = as_int(inputs.get("trestbps"))
        chol = as_int(inputs.get("chol"))
        fbs_bool = parse_bool_like(inputs.get("fbs"))
        restecg = norm(inputs.get("restecg"))
        thalach = as_int(inputs.get("thalach"))
        exang = parse_bool_like(inputs.get("exang"))
        oldpeak = as_float(inputs.get("oldpeak"))
        slope = norm(inputs.get("slope"))
        ca = as_int(inputs.get("ca"))
        thal = norm(inputs.get("thal"))

        # -- Indicator computations (conservative) --
        # Vessel flag: any vessel colored (ca >= 1)
        vessel_flag = (ca is not None and ca >= 1)

        # Typical chest pain (conservative match): explicit "typical" token or common numeric code '1'
        typical_cp = cp_is_typical(cp)

        # Exercise-induced ischemia: exercise angina with a low achieved max heart rate.
        # We treat exang alone as weak; require low thalach to count here.
        exang_ischemia = False
        if exang is True and thalach is not None:
            # low threshold chosen conservatively: < 120 bpm suggests limited exertional capacity
            if thalach < 120:
                exang_ischemia = True

        # Abnormal resting ECG: look for ST-T abnormalities or LVH tokens
        restecg_abnormal = contains_token(restecg, ("st-t", "st t", "st-t wave", "st-t wave abnormality", "abnormal", "left ventricular hypertrophy", "lvh"))

        # Abnormal thal only counts when at least one vessel is flagged (conservative)
        abnormal_thal = contains_token(thal, ("fixed", "reversible", "defect"))
        abnormal_thal_and_vessel = abnormal_thal and vessel_flag

        # Large ST depression contributes but should not by itself trigger 'yes'
        oldpeak_high = (oldpeak is not None and oldpeak > 3.0)

        # Compose indicator score
        indicators = 0
        if vessel_flag:
            indicators += 1
        if typical_cp:
            indicators += 1
        if exang_ischemia:
            indicators += 1
        if restecg_abnormal:
            indicators += 1
        if abnormal_thal_and_vessel:
            indicators += 1
        # Count oldpeak_high as an indicator, but because final rule requires >=2,
        # oldpeak alone will not trigger "yes".
        if oldpeak_high:
            indicators += 1

        # Final conservative decision:
        # - Need at least two indicator points to return "yes".
        # This prevents single-factor positives (e.g., ca=2 alone) from forcing 'yes'.
        strong_yes = (indicators >= 2)

        answer = "yes" if strong_yes else "no"
        return dspy.Prediction(answer=answer)

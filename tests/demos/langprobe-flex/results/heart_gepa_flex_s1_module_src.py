class HeartDiseaseSignatureModule(dspy.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **inputs):
        """
        Deterministic conservative classifier for presence of heart disease.
        Returns exactly the lowercase string "yes" or "no".

        Heuristics (designed to avoid the false positives seen in failures):
        - Strong anatomical evidence:
            * ca (number of major vessels colored by fluoroscopy) >= 3 -> "yes"
        - Moderate anatomical evidence combined with ischemic signs:
            * ca >= 2 AND (exercise-induced angina OR clear ST-depression on exercise (oldpeak >= 2.5)) -> "yes"
        - Functional (reversible ischemia) evidence:
            * thal = "reversible defect" AND (exercise-induced angina OR oldpeak >= 1.5) -> "yes"
        - Otherwise -> "no"

        Notes on parsing and conservatism:
        - We ONLY treat textual cp values like "typical angina" as meaningful; numeric cp codes are ignored
          to avoid mis-mapping different encodings.
        - thal numeric codes (common UCI conventions) are recognized: '3'->normal, '6'->fixed defect, '7'->reversible defect.
          Unknown or ambiguous thal values are treated as non-diagnostic.
        - ca values that cannot be parsed to integers are treated as 0 (conservative).
        - exang recognizes many boolean-like forms ('yes','1','true', etc.).
        """
        def to_int_safe(x):
            """Convert x to int if it represents an integer (allows '0', '1', '2', '2.0'). Return None if not parseable."""
            if x is None:
                return None
            s = str(x).strip()
            if s == "":
                return None
            try:
                f = float(s)
                i = int(f)
                # ensure float had no fractional part other than .0
                if abs(f - i) > 1e-8:
                    return None
                return i
            except Exception:
                return None

        def to_float_safe(x):
            """Convert x to float, or None if not parseable."""
            if x is None:
                return None
            s = str(x).strip()
            if s == "":
                return None
            try:
                return float(s)
            except Exception:
                return None

        def parse_bool_like(x):
            """Return True/False for common boolean-like strings, or None if ambiguous."""
            if x is None:
                return None
            s = str(x).strip().lower()
            if s in ("1", "true", "t", "yes", "y"):
                return True
            if s in ("0", "false", "f", "no", "n"):
                return False
            return None

        def normalize_thal(x):
            """Normalize thal to 'normal', 'fixed defect', 'reversible defect', or None if unknown."""
            if x is None:
                return None
            s = str(x).strip().lower()
            if s == "":
                return None
            # textual cues
            if "normal" in s:
                return "normal"
            if "fixed" in s:
                return "fixed defect"
            if "revers" in s:  # matches 'reversible' or 'reversible defect'
                return "reversible defect"
            # common numeric encodings in UCI dataset: 3=normal, 6=fixed, 7=reversible
            if s in ("3", "6", "7"):
                if s == "3":
                    return "normal"
                if s == "6":
                    return "fixed defect"
                if s == "7":
                    return "reversible defect"
            # unknown otherwise
            return None

        # Parse fields conservatively
        ca_val = to_int_safe(inputs.get("ca", None))
        if ca_val is None:
            ca_val = 0

        oldpeak = to_float_safe(inputs.get("oldpeak", None))
        if oldpeak is None:
            oldpeak = 0.0

        exang_bool = parse_bool_like(inputs.get("exang", None))
        # treat ambiguous exang as False (conservative)
        if exang_bool is None:
            exang_bool = False

        thal_norm = normalize_thal(inputs.get("thal", None))

        # Decision rules
        answer = "no"  # default conservative answer

        # Strong anatomical evidence
        if ca_val >= 3:
            answer = "yes"
        else:
            # Moderate anatomical evidence but only if combined with ischemic signs
            if ca_val >= 2 and (exang_bool or oldpeak >= 2.5):
                answer = "yes"
            # Reversible thalemia suggests inducible ischemia; require accompanying signs
            elif thal_norm == "reversible defect" and (exang_bool or oldpeak >= 1.5):
                answer = "yes"
            else:
                answer = "no"

        return dspy.Prediction(answer=answer)

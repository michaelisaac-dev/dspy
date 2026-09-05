class StringSignatureModule(dspy.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **inputs):
        """
        Conservative deterministic classifier for whether an employer called back.

        Strategy (conservative because callbacks are rare):
        - Normalize inputs into booleans/ints safely.
        - Compute a compact scored rubric where a few signals are strong (resume quality,
          special_skills, honors, very high experience) and many are weak supportive signals.
        - Require BOTH: (score >= threshold) AND (at least one independent strong signal).
          This prevents marginal profiles with many small positives from predicting 'yes'.
        - Do not call an LLM: this is a compact, interpretable heuristic.
        """

        def norm_bool(val):
            """Normalize common truthy/falsy encodings to True/False/None."""
            if val is None:
                return None
            s = str(val).strip().lower()
            if s in ("1", "yes", "true", "y", "t"):
                return True
            if s in ("0", "no", "false", "n", "f"):
                return False
            return None

        def to_int(val, default=0):
            """Convert strings like '4' or '4.0' to int when possible, otherwise default."""
            try:
                return int(float(str(val).strip()))
            except Exception:
                return default

        # Read and normalize inputs
        resume_quality = str(inputs.get("resume_quality", "")).strip().lower()
        has_email = norm_bool(inputs.get("has_email_address"))
        employment_holes_raw = str(inputs.get("employment_holes", "")).strip().lower()
        # employment_holes: '1' means gaps present, '0' means none
        if employment_holes_raw in ("1", "0"):
            employment_holes = (employment_holes_raw == "1")
        else:
            employment_holes = None

        years_experience = to_int(inputs.get("years_experience", 0))
        years_college = to_int(inputs.get("years_college", 0))

        college_degree = norm_bool(inputs.get("college_degree"))
        honors = norm_bool(inputs.get("honors"))
        worked_during_school = norm_bool(inputs.get("worked_during_school"))
        computer_skills = norm_bool(inputs.get("computer_skills"))
        special_skills = norm_bool(inputs.get("special_skills"))
        volunteer = norm_bool(inputs.get("volunteer"))
        military = norm_bool(inputs.get("military"))

        # Scoring rubric (conservative)
        # Strong signals: special_skills, honors, very high experience (>=10)
        # Weak/moderate signals: email, no employment gaps, college degree, etc.
        score = 0.0

        # Resume quality: high gives meaningful boost, low penalizes
        if resume_quality == "high":
            score += 2.0
        elif resume_quality == "low":
            score -= 2.0

        # Contactability
        if has_email is True:
            score += 0.5

        # Employment gaps: no gaps is a small positive, gaps are a bigger negative
        if employment_holes is False:  # no gaps
            score += 0.5
        elif employment_holes is True:  # gaps present
            score -= 1.0

        # Education and recognitions
        if college_degree is True:
            score += 0.5
        if honors is True:
            score += 0.75  # honors is a relatively strong positive

        # Work during school, volunteer, military, computer skills are weak positives
        if worked_during_school is True:
            score += 0.25
        if computer_skills is True:
            score += 0.25
        if special_skills is True:
            score += 1.5  # special skills treated as a strong signal
        if volunteer is True:
            score += 0.25
        if military is True:
            score += 0.25

        # Experience weighting (moderate to strong for long tenure)
        if years_experience >= 10:
            score += 2.0
        elif years_experience >= 5:
            score += 0.75
        elif years_experience >= 2:
            score += 0.25

        # College years small bonus
        if years_college >= 4:
            score += 0.25

        # Conservative decision rule:
        # - Require numeric threshold AND an independent strong signal.
        threshold = 5.0
        strong_signal = (
            (special_skills is True)
            or (honors is True)
            or (years_experience >= 10)
        )

        pred_yes = (score >= threshold) and strong_signal

        # Final output
        received_callback = "yes" if pred_yes else "no"
        return dspy.Prediction(received_callback=received_callback)

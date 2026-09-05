class StringSignatureModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # Deterministic, conservative classifier — no LM needed.

    def forward(self, **inputs):
        """
        Predict whether the employer called the applicant back ('yes' or 'no') using a
        conservative rule-based decision procedure.

        Rationale for the rules (based on feedback about a low callback base rate ~7.5%):
        - Only predict 'yes' when there is a strong, combined signal (resume quality + contactability
          + one or more strong qualifications). We avoid saying 'yes' for marginal cases.
        - Treat special skills and a completed college degree as strong positive signals.
        - Long experience can be strong, but gaps in employment (employment_holes) make a callback much
          less likely unless other strong signals are present and there are no gaps.
        - Honors are only meaningful if a degree is present.
        - Always require a working email address (has_email_address) to consider a callback possible.

        The decision rule implemented:
        1. If resume_quality != 'high' -> 'no' (very conservative).
        2. Require has_email_address to be present/true.
        3. Then require at least one of:
           A) special_skills AND college_degree
           B) special_skills AND years_experience >= 15 AND NO employment_holes
           C) college_degree AND honors AND computer_skills
        If these conditions are met -> 'yes', otherwise -> 'no'.

        This rule set was crafted to reduce false positives while still allowing
        high-confidence cases to get a 'yes'.
        """
        def to_int(x, default=0):
            try:
                return int(x)
            except Exception:
                return default

        def is_one(x):
            if x is None:
                return False
            s = str(x).strip().lower()
            return s in ("1", "yes", "true", "y", "t")

        def is_high(x):
            if x is None:
                return False
            return str(x).strip().lower() == "high"

        # Safe reads
        resume_quality = inputs.get("resume_quality")
        years_experience = to_int(inputs.get("years_experience"), default=0)
        college_degree = is_one(inputs.get("college_degree"))
        honors = is_one(inputs.get("honors"))
        computer_skills = is_one(inputs.get("computer_skills"))
        special_skills = is_one(inputs.get("special_skills"))
        volunteer = is_one(inputs.get("volunteer"))
        worked_during_school = is_one(inputs.get("worked_during_school"))
        employment_holes = is_one(inputs.get("employment_holes"))  # True means gaps present
        has_email = is_one(inputs.get("has_email_address"))

        # Rule 1: must be high resume quality to consider a callback (conservative baseline).
        if not is_high(resume_quality):
            return dspy.Prediction(received_callback="no")

        # Rule 2: must have an email address to be contactable.
        if not has_email:
            return dspy.Prediction(received_callback="no")

        # Strong positive conditions (need at least one)
        cond_A = special_skills and college_degree
        cond_B = special_skills and (years_experience >= 15) and (not employment_holes)
        cond_C = college_degree and honors and computer_skills

        if cond_A or cond_B or cond_C:
            return dspy.Prediction(received_callback="yes")
        else:
            return dspy.Prediction(received_callback="no")

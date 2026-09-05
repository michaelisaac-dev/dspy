class StringSignatureModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # Deterministic, rule-based classifier — no LM required.

    def forward(self, **inputs):
        """
        Predict whether the employer called this applicant back after receiving the resume.
        Returns 'yes' or 'no'.

        Design goals (from failure analysis):
        - Be conservative: callbacks are relatively rare (~8%), so require multiple
          moderately strong positive signals before predicting 'yes'.
        - Rely only on applicant / resume signals that reflect likely employer preferences
          (experience, honors, special skills, resume quality, contactability).
        - Do not use protected attributes (race, gender) to avoid introducing bias.
        - Parse noisy inputs robustly (strings like '1','yes','true', numeric text for years).
        """

        import re

        def parse_int_first(s: str) -> int:
            """Return the first integer found in the string s, or 0 if none."""
            if s is None:
                return 0
            s = str(s)
            m = re.search(r"\d+", s)
            if not m:
                return 0
            try:
                return int(m.group(0))
            except Exception:
                return 0

        def is_positive_flag(s: str) -> bool:
            """
            Interpret common binary encodings: '1','yes','true','y','t' -> True;
            empty/None/'0'/'no'/'false' -> False.
            """
            if s is None:
                return False
            s = str(s).strip().lower()
            return s in ("1", "yes", "true", "y", "t")

        # Read inputs (use .get so missing fields are handled gracefully)
        resume_quality = str(inputs.get("resume_quality", "")).strip().lower()
        years_experience_raw = inputs.get("years_experience", "")
        years_experience = parse_int_first(years_experience_raw)
        college_degree = inputs.get("college_degree", "")
        honors = inputs.get("honors", "")
        worked_during_school = inputs.get("worked_during_school", "")
        computer_skills = inputs.get("computer_skills", "")
        special_skills = inputs.get("special_skills", "")
        volunteer = inputs.get("volunteer", "")
        military = inputs.get("military", "")
        employment_holes = inputs.get("employment_holes", "")
        has_email_address = inputs.get("has_email_address", "")

        # Conservative scoring: only a few signals give substantial positive weight.
        score = 0.0

        # Strong signals
        if is_positive_flag(honors):
            score += 2.0  # honors strongly increases callback likelihood
        if is_positive_flag(special_skills):
            score += 1.5  # special/technical skills are valuable

        # Resume quality — helpful but not decisive by itself
        if resume_quality == "high":
            score += 1.0
        elif resume_quality == "low":
            score -= 1.0

        # Experience tiers (moderate signal)
        if years_experience >= 8:
            score += 1.0
        elif 4 <= years_experience <= 7:
            score += 0.25
        elif 1 <= years_experience <= 3:
            score -= 0.5
        else:
            # 0 or unknown -> no contribution
            pass

        # Contactability: having an email is modestly positive
        if is_positive_flag(has_email_address):
            score += 0.5
        else:
            # missing email lowers chance a little
            score -= 0.25

        # Education and basic skills: small positive contributions
        if is_positive_flag(college_degree):
            score += 0.25
        if is_positive_flag(computer_skills):
            score += 0.25

        # Other minor positive signals
        if is_positive_flag(volunteer):
            score += 0.25
        if is_positive_flag(worked_during_school):
            score += 0.2

        # Military and employment gaps: treat cautiously (small or neutral)
        # Employment gaps in this dataset are not strongly negative, so do not penalize heavily.
        if is_positive_flag(military):
            score += 0.0
        if is_positive_flag(employment_holes):
            score += 0.25

        # Final decision: require a fairly strong score to predict 'yes'.
        # Threshold chosen to keep predictions conservative (few false positives).
        prediction = "yes" if score >= 4.0 else "no"

        return dspy.Prediction(received_callback=prediction)

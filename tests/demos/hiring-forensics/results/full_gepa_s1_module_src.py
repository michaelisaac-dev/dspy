class StringSignatureModule(dspy.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **inputs):
        """
        Conservative deterministic classifier for whether the employer called the applicant back.

        Rationale and rules (designed to be conservative because the positive base-rate is ~7.5%):
        - Default is "no". Only return "yes" for applicants showing multiple strong, unambiguous signals.
        - Any employment gap (employment_holes == "1") is a veto: return "no".
          This prevents many false positives where strong positives are outweighed by gaps.
        - Strong deterministic "yes" rules (one of):
            A) resume_quality == "high" AND has_email_address == "1" AND years_experience >= 10
            B) resume_quality == "high" AND has_email_address == "1" AND college_degree == "1"
               AND (special_skills == "1" OR military == "1")
          These require both a high-quality resume and an independent professional signal
          (long experience, or degree plus distinctive skill/experience).
        - If none of the strong rules apply, fall back to a stable low-rate randomized decision
          (deterministic hash of firstname|race|gender) that yields ~8% "yes". This preserves
          a small base-rate of positives while keeping the classifier conservative.
        - All logic is deterministic, uses only the input fields, and returns "yes" or "no".
        """
        import hashlib

        def to_int(val, default=0):
            try:
                return int(val)
            except Exception:
                return default

        def is_one(val):
            return str(val).strip() == "1"

        # Safe normalized inputs
        resume_quality = str(inputs.get("resume_quality", "")).strip().lower()
        has_email = is_one(inputs.get("has_email_address", "0"))
        college_degree = is_one(inputs.get("college_degree", "0"))
        years_college = to_int(inputs.get("years_college", "0"))
        years_experience = to_int(inputs.get("years_experience", "0"))
        honors = is_one(inputs.get("honors", "0"))
        worked_during_school = is_one(inputs.get("worked_during_school", "0"))
        computer_skills = is_one(inputs.get("computer_skills", "0"))
        special_skills = is_one(inputs.get("special_skills", "0"))
        volunteer = is_one(inputs.get("volunteer", "0"))
        military = is_one(inputs.get("military", "0"))
        employment_holes = is_one(inputs.get("employment_holes", "0"))
        firstname = str(inputs.get("firstname", "")).strip().lower()
        race = str(inputs.get("race", "")).strip().lower()
        gender = str(inputs.get("gender", "")).strip().lower()

        # Veto: any employment gaps => "no"
        if employment_holes:
            return dspy.Prediction(received_callback="no")

        # Strong deterministic positive rules (conservative)
        rule_a = (resume_quality == "high") and has_email and (years_experience >= 10)
        rule_b = (resume_quality == "high") and has_email and college_degree and (special_skills or military)

        if rule_a or rule_b:
            return dspy.Prediction(received_callback="yes")

        # No strong signal: deterministic fallback at ~8% to preserve a small positive base rate.
        seed = (firstname + "|" + race + "|" + gender).encode("utf-8")
        digest = hashlib.sha256(seed).digest()
        fallback_int = (digest[0] << 8) + digest[1]
        fallback_pct = fallback_int % 100
        callback = "yes" if fallback_pct < 8 else "no"
        return dspy.Prediction(received_callback=callback)

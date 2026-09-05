class StringSignatureModule(dspy.Module):
    """
    Predict whether the employer called this applicant back after receiving the resume.

    Design notes:
    - This is a conservative deterministic heuristic (no LM). The task's feedback
      showed the model was producing too many false "yes" predictions. Only a small
      fraction of applicants receive callbacks, so this classifier raises the bar:
        * Strong positive signals (excellent resume, long experience) are required
          and must combine to a high aggregated score.
        * Moderate positives (degree, contactability, basic skills) help but are
          not sufficient on their own.
        * Certain roles (general "sales", "retail") are treated cautiously and receive
          a penalty because they produced many false positives in failures.
        * Employment gaps produce a strong penalty.
    - The decision rule was tuned so that only applicants with multiple strong signals
      reach the "yes" threshold (threshold tuned upward after failures).
    """

    def __init__(self):
        super().__init__()
        # Pure Python heuristic: no predictors required.

    def forward(self, **inputs):
        # Helpers
        def norm(s: object) -> str:
            if s is None:
                return ""
            try:
                return str(s).strip().lower()
            except Exception:
                return ""

        def to_int(s: object, default: int = 0) -> int:
            s2 = norm(s)
            try:
                return int(float(s2))
            except Exception:
                return default

        def is_truthy(s: object) -> bool:
            return norm(s) in ("1", "yes", "true", "y", "t")

        # Read and normalize inputs
        job_type = norm(inputs.get("job_type", ""))
        job_industry = norm(inputs.get("job_industry", ""))
        resume_quality = norm(inputs.get("resume_quality", ""))
        college_degree = inputs.get("college_degree", "")
        has_email = inputs.get("has_email_address", "")
        computer_skills = inputs.get("computer_skills", "")
        special_skills = inputs.get("special_skills", "")
        worked_during_school = inputs.get("worked_during_school", "")
        volunteer = inputs.get("volunteer", "")
        employment_holes = inputs.get("employment_holes", "")
        years_experience = to_int(inputs.get("years_experience", 0))
        years_college = to_int(inputs.get("years_college", 0))

        # Scoring constants (chosen to be conservative; require multiple strong signals)
        score = 0.0

        # Resume quality: high is a strong positive signal.
        if resume_quality == "high":
            score += 3.0
        # low or unknown -> no bonus

        # Contactability is necessary but not decisive
        if is_truthy(has_email):
            score += 1.0

        # Experience: larger step for long tenures
        if years_experience >= 15:
            score += 4.0
        elif years_experience >= 10:
            score += 3.0
        elif years_experience >= 5:
            score += 1.5
        elif years_experience >= 2:
            score += 0.5
        # <2 years -> no experience bonus

        # Education signals (helpful but modest)
        if is_truthy(college_degree) or years_college >= 2:
            score += 0.75

        # Skills
        if is_truthy(computer_skills):
            score += 0.5
        if is_truthy(special_skills):
            score += 0.75

        # Work/volunteer signals (small positive)
        if is_truthy(worked_during_school):
            score += 0.25
        if is_truthy(volunteer):
            score += 0.25

        # Employment gaps strongly penalize callback likelihood
        if is_truthy(employment_holes):
            score -= 2.5

        # Role- and industry-specific adjustments:
        jt = job_type.replace(" ", "_")
        ji = job_industry

        # Manager roles are more likely to get callbacks, but require support from other signals
        if "manager" in jt:
            score += 1.0

        # Secretary/administrative roles: small boost but not large
        if jt in ("secretary", "administrative_assistant", "administrative"):
            score += 0.5

        # Be cautious about sales and retail roles (observed false positives).
        # Apply a penalty for generic sales roles and stronger penalty when the industry suggests retail.
        if "sales" in jt or "sales" in ji:
            score -= 1.5
            if "retail" in ji or "wholesale" in ji or "retail" in jt:
                score -= 1.5  # additional penalty for retail/wholesale context

        # Final decision: require a high aggregated score to predict "yes".
        # This threshold is intentionally high to reduce false positives observed in feedback.
        threshold = 9.0

        received_callback = "yes" if score >= threshold else "no"

        # Defensive: ensure returned value is exactly "yes" or "no"
        if received_callback not in ("yes", "no"):
            received_callback = "no"

        return dspy.Prediction(received_callback=received_callback)

class StringSignatureModule(dspy.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **inputs):
        """
        Deterministic heuristic classifier that estimates the callback probability
        by averaging empirical callback rates for the provided fields (derived
        from the training-population summary that was provided in feedback).
        Returns 'yes' only when the estimated probability meets a conservative
        threshold (0.09 = 9%), otherwise 'no'.

        Rules:
        - Use only the per-field empirical rates listed below when an input's
          categorical value matches one of the known categories.
        - For numeric years_experience, bucket into: '<=3', '4-7', '8+'.
        - If no known-field matches are available, default to 'no' (conservative).
        - Output is the string 'yes' or 'no'.
        """
        # Normalized mappings of field value -> observed callback rate (as decimal)
        rates = {
            "job_city": {"chicago": 0.069, "boston": 0.100},
            "job_industry": {
                "other_service": 0.088,
                "business_and_personal_service": 0.091,
                "wholesale_and_retail_trade": 0.070,
                "manufacturing": 0.055,
                "finance_insurance_real_estate": 0.086,
                "transportation_communication": 0.127,
            },
            "job_type": {
                "secretary": 0.085,
                "retail_sales": 0.085,
                "sales_rep": 0.074,
                "manager": 0.066,
                "clerical": 0.116,
                "supervisor": 0.068,
            },
            "job_equal_opp_employer": {"0": 0.084, "1": 0.079},
            "job_fed_contractor": {"0": 0.080, "unknown": 0.085, "1": 0.092},
            "years_college": {"4": 0.081, "3": 0.089, "2": 0.097, "0": 0.032},
            "college_degree": {"1": 0.081, "0": 0.088},
            "honors": {"0": 0.079, "1": 0.158},
            "worked_during_school": {"1": 0.074, "0": 0.094},
            # years_experience handled via bucketing below
            "computer_skills": {"1": 0.079, "0": 0.101},
            "special_skills": {"0": 0.062, "1": 0.125},
            "volunteer": {"0": 0.082, "1": 0.085},
            "military": {"0": 0.085, "1": 0.059},
            "employment_holes": {"0": 0.068, "1": 0.102},
            "has_email_address": {"0": 0.077, "1": 0.089},
            "resume_quality": {"high": 0.090, "low": 0.076},
            "race": {"black": 0.062, "white": 0.104},
            "gender": {"f": 0.086, "m": 0.070},
        }

        def bucket_years_experience(val: str) -> str | None:
            """Bucket a years_experience input into one of the known categories."""
            if val is None:
                return None
            s = str(val).strip().lower()
            # If already in bucket form, return directly
            if s in {"4-7", "8+", "<=3"}:
                return s
            # Try parse integer
            try:
                n = int(float(s))
                if n <= 3:
                    return "<=3"
                if 4 <= n <= 7:
                    return "4-7"
                return "8+"
            except Exception:
                return None

        # conservative default base probability (used only if no matched fields)
        base_prob = 0.08
        matched_probs = []

        # For each field for which we have empirical rates, try to find a match.
        for field, mapping in rates.items():
            raw = inputs.get(field)
            if raw is None:
                continue
            val = str(raw).strip().lower()
            # direct match for mapped categories
            if val in mapping:
                matched_probs.append(mapping[val])
            # Some inputs are numeric strings that should match keys exactly (handled above).
            # No else: skip unmatched values to avoid making unsupported assumptions.

        # Handle years_experience specially (bucketing then lookup)
        ye_raw = inputs.get("years_experience")
        ye_bucket = bucket_years_experience(ye_raw)
        if ye_bucket is not None:
            ye_map = {"4-7": 0.074, "8+": 0.104, "<=3": 0.054}
            if ye_bucket in ye_map:
                matched_probs.append(ye_map[ye_bucket])

        # If we matched at least one field, average their empirical rates.
        if matched_probs:
            avg_prob = sum(matched_probs) / len(matched_probs)
        else:
            avg_prob = base_prob

        # Conservative decision threshold: require estimated prob >= 9% to predict 'yes'.
        threshold = 0.09
        received_callback = "yes" if avg_prob >= threshold else "no"
        return dspy.Prediction(received_callback=received_callback)

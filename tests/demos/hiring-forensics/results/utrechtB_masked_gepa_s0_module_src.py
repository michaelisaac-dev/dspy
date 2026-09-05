class StringSignatureModule(dspy.Module):
    """
    Predict whether the company decided to hire the candidate.

    Deterministic, rule-based classifier tuned to reflect the training-set
    tendencies (no LM calls). The rules encode strong positive/negative
    signals observed in the dataset and a small set of exception rules
    that correct common false positives seen in the feedback.

    Inputs (all strings):
      - sport, university_grade, debateclub, programming_exp, international_exp,
        entrepeneur_exp, languages, exact_study, degree, exact_study

    Output:
      - decision: 'yes' or 'no'
    """
    def __init__(self):
        super().__init__()

    def forward(self, **inputs):
        # Helpers ---------------------------------------------------------
        def is_true(val: str) -> bool:
            """Robust boolean interpretation for string-ish inputs."""
            if val is None:
                return False
            v = str(val).strip().lower()
            return v in ("true", "t", "yes", "y", "1")

        def to_int(val):
            """Parse an int from a string-like value; return None if not parseable."""
            try:
                return int(str(val).strip())
            except Exception:
                return None

        def grade_bucket(grade_val):
            """
            Map a numeric university grade to a coarse bucket used in the rules:
            returns '50s', '60s', '70s', or None if grade is missing/unparsable.
            """
            g = to_int(grade_val)
            if g is None:
                return None
            if 50 <= g <= 59:
                return "50s"
            if 60 <= g <= 69:
                return "60s"
            if 70 <= g <= 79:
                return "70s"
            # other ranges are treated as None (no strong signal)
            return None

        # Read and normalize inputs --------------------------------------
        sport = (inputs.get("sport") or "").strip().lower()
        university_grade = inputs.get("university_grade")
        debateclub = inputs.get("debateclub")
        programming_exp = inputs.get("programming_exp")
        international_exp = inputs.get("international_exp")
        entrepeneur_exp = inputs.get("entrepeneur_exp")  # note the dataset's spelling
        languages = inputs.get("languages")
        exact_study = inputs.get("exact_study")
        degree = (inputs.get("degree") or "").strip().lower()

        debate_flag = is_true(debateclub)
        entrepreneur_flag = is_true(entrepeneur_exp)
        programming_flag = is_true(programming_exp)
        international_flag = is_true(international_exp)
        exact_study_flag = is_true(exact_study)
        lang_count = to_int(languages)
        grade_bucket_val = grade_bucket(university_grade)

        # Decision rules (ordered, deterministic) ------------------------
        # Core strong positive: debate club membership (very high hire rate).
        if debate_flag:
            decision = "yes"
        else:
            # Entrepreneur experience is a generally strong positive, but the
            # training-set contains counterexamples where a low grade + bachelor
            # degree override that signal. Encode that exception to avoid common
            # false positives seen in feedback.
            if entrepreneur_flag:
                # Exception / veto: if candidate is in the 60s grade bucket AND
                # has a bachelor degree, treat as not-hired despite entrepreneur flag.
                if grade_bucket_val == "60s" and degree.startswith("b"):
                    decision = "no"
                else:
                    decision = "yes"
            else:
                # Languages: speaking multiple languages is a positive signal.
                if lang_count is not None and lang_count >= 2:
                    decision = "yes"
                else:
                    # All remaining patterns default to the more-common 'no'.
                    # (Other fields like programming_exp True, exact_study True,
                    # and certain degree/grade combos are weak negatives already
                    # reflected by the default.)
                    decision = "no"

        # Final normalization and return --------------------------------
        decision = "yes" if str(decision).strip().lower() == "yes" else "no"
        return dspy.Prediction(decision=decision)

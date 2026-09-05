class StringSignatureModule(dspy.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **inputs):
        """
        Deterministic rule + data-driven scorer that encodes the HIRE RATES BY FIELD.
        No LMs used.

        Revisions made to fix failures:
        - Keep the conservative numeric scoring and the debate+positive override.
        - Tighten the "multiple high-rate fields" rule: require 3+ high-rate fields
          AND at least one clear behavioral/experience signal (debate membership,
          entrepreneurial experience, or speaking >= 2 languages). This prevents
          demographic/sport/university-only clusters from triggering hire (fixes
          Example 2).
        - Defensive parsing and defaults for all fields.
        """

        def norm_bool(x):
            if isinstance(x, bool):
                return x
            if x is None:
                return False
            s = str(x).strip().lower()
            return s in ("true", "1", "yes", "y", "t")

        # Per-field hire-rate lookup from the training pool
        gender_rates = {"male": 0.477, "female": 0.116}
        age_rates = {"older": 0.332, "younger": 0.292}
        nationality_rates = {"dutch": 0.315, "belgian": 0.318, "german": 0.327}
        sport_rates = {
            "football": 0.466, "rugby": 0.600, "swimming": 0.176,
            "tennis": 0.485, "chess": 0.018, "cricket": 0.118
        }
        university_grade_bands = {"50s": 0.452, "60s": 0.271, "70s": 0.232}
        debate_rates = {True: 0.797, False: 0.159}
        programming_rates = {True: 0.163, False: 0.385}
        international_rates = {True: 0.284, False: 0.327}
        entrepeneur_rates = {True: 0.775, False: 0.163}
        languages_rates = {0: 0.0, 1: 0.021, 2: 0.626, 3: 0.946}
        exact_study_rates = {True: 0.153, False: 0.461}
        degree_rates = {"bachelor": 0.272, "master": 0.372, "phd": 0.294}

        # Safe extractors / mappers
        def get_gender_rate(val):
            if val is None:
                return 0.31
            return gender_rates.get(str(val).strip().lower(), 0.31)

        def get_age_rate(val):
            try:
                a = int(str(val).strip())
                return age_rates["older"] if a > 25 else age_rates["younger"]
            except Exception:
                return 0.31

        def get_nationality_rate(val):
            if val is None:
                return 0.31
            return nationality_rates.get(str(val).strip().lower(), 0.31)

        def get_sport_rate(val):
            if val is None:
                return 0.31
            return sport_rates.get(str(val).strip().lower(), 0.31)

        def get_university_rate(val):
            # university_grade expected on 0-100 scale; map to tens band
            try:
                g = int(float(str(val).strip()))
                if 50 <= g <= 59:
                    return university_grade_bands["50s"]
                if 60 <= g <= 69:
                    return university_grade_bands["60s"]
                if 70 <= g <= 79:
                    return university_grade_bands["70s"]
                # If outside these bands return a reasonable base
                return 0.31
            except Exception:
                return 0.31

        def get_bool_rate(mapping, val):
            b = norm_bool(val)
            return mapping.get(b, 0.31)

        def get_languages_rate(val):
            try:
                n = int(str(val).strip())
                # clamp to known keys 0..3
                if n < 0:
                    n = 0
                if n > 3:
                    n = 3
                return languages_rates.get(n, 0.31)
            except Exception:
                return 0.31

        def get_degree_rate(val):
            if val is None:
                return 0.31
            return degree_rates.get(str(val).strip().lower(), 0.31)

        # Gather per-field rates (for the fields in the signature)
        per_field_rates = []
        per_field_rates.append(get_gender_rate(inputs.get("gender")))
        per_field_rates.append(get_age_rate(inputs.get("age")))
        per_field_rates.append(get_nationality_rate(inputs.get("nationality")))
        per_field_rates.append(get_sport_rate(inputs.get("sport")))
        per_field_rates.append(get_university_rate(inputs.get("university_grade")))
        per_field_rates.append(get_bool_rate(debate_rates, inputs.get("debateclub")))
        per_field_rates.append(get_bool_rate(programming_rates, inputs.get("programming_exp")))
        per_field_rates.append(get_bool_rate(international_rates, inputs.get("international_exp")))
        per_field_rates.append(get_bool_rate(entrepeneur_rates, inputs.get("entrepeneur_exp")))
        per_field_rates.append(get_languages_rate(inputs.get("languages")))
        per_field_rates.append(get_bool_rate(exact_study_rates, inputs.get("exact_study")))
        per_field_rates.append(get_degree_rate(inputs.get("degree")))

        # Base score: mean of available per-field rates (fallback to 0.31)
        if len(per_field_rates) == 0:
            base = 0.31
        else:
            base = sum(per_field_rates) / len(per_field_rates)

        # Conservative adjustments (smaller absolute effects)
        debate = norm_bool(inputs.get("debateclub"))
        entre = norm_bool(inputs.get("entrepeneur_exp"))
        prog = norm_bool(inputs.get("programming_exp"))
        exact = norm_bool(inputs.get("exact_study"))
        try:
            langs = int(str(inputs.get("languages")).strip())
        except Exception:
            langs = 0
        # Parse numeric university grade defensively
        try:
            uni_grade_val = int(float(str(inputs.get("university_grade")).strip()))
        except Exception:
            uni_grade_val = None
        degree_raw = (inputs.get("degree") or "")
        degree_norm = str(degree_raw).strip().lower()

        score = base

        # Debate membership is a very strong positive signal in the dataset.
        # Give a moderate additive boost (and use the stronger override rule below).
        if debate:
            score += 0.18

        # Entrepreneurial experience is strongly positive in the data.
        if entre:
            score += (0.12 if debate else 0.06)

        # Programming experience correlates with lower hire rates; modest penalty.
        if prog:
            score -= 0.08

        # Exact-study correlates with lower hires; modest penalty.
        if exact:
            score -= 0.06

        # Language bonuses: 3 languages very positive, 2 languages somewhat positive.
        if langs == 3:
            score += 0.12
        elif langs == 2:
            score += 0.06

        # Clip score to [0,1]
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0

        # High-priority rule to reflect decisive debate membership:
        strong_positive_indicators = (
            entre
            or (isinstance(langs, int) and langs >= 2)
            or (isinstance(uni_grade_val, int) and uni_grade_val >= 50)
            or (degree_norm in ("master", "phd"))
        )
        if debate and strong_positive_indicators:
            return dspy.Prediction(decision="yes")

        # Count independent high-rate fields (per-field rate >= 0.45).
        high_rate_count = 0
        for r in per_field_rates:
            try:
                if float(r) >= 0.45:
                    high_rate_count += 1
            except Exception:
                continue

        # Revised rule:
        # If multiple independent fields are strongly positive, require at least one
        # clear behavioral/experience signal (debate, entrepreneur, or >=2 languages)
        # to predict 'yes'. This avoids demographic-only clusters incorrectly triggering hire.
        if high_rate_count >= 3 and (debate or entre or (isinstance(langs, int) and langs >= 2)):
            return dspy.Prediction(decision="yes")

        # Otherwise use the numeric threshold.
        decision = "yes" if score >= 0.5 else "no"
        return dspy.Prediction(decision=decision)

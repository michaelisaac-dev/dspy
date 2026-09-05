class StringSignatureModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # Deterministic, rule-based module: no LLM predictors required.

    def forward(self, **inputs):
        """
        Decide 'yes' or 'no' deterministically using the training-pool hire rates
        encoded below. The decision process is:

        1. For every input field present and recognized, look up its observed hire
           rate from the training statistics (the domain tables below).
        2. Compute a weighted average of the included rates. The base rate (overall
           hire rate) is always included with weight 1.
        3. Fields that show especially strong, high-variance signals in the
           training data are given larger weight so they impact the decision more:
           debateclub, entrepeneur_exp, and languages receive weight 3; other
           fields use weight 1.
        4. If the weighted average >= base_rate, return "yes", otherwise "no".

        This deterministic rule set was chosen to reflect the dataset-level
        signals (strong positive effects from debate/entrepreneurship/multiple
        languages) while remaining fully deterministic and interpretable.
        """
        def parse_bool(s: str) -> bool:
            if s is None:
                return False
            s = str(s).strip().lower()
            return s in ("true", "1", "yes", "y", "t")

        def safe_int(s, default=None):
            try:
                # handle floats in string form as well as ints
                return int(float(str(s).strip()))
            except Exception:
                return default

        # --- domain hire-rate tables (from training-pool statistics) ---
        base_rate = 0.31  # overall base hire rate

        gender_rates = {
            "male": 0.477,
            "female": 0.116,
        }
        age_rates = {
            "le_25": 0.292,
            "gt_25": 0.332,
        }
        nationality_rates = {
            "dutch": 0.315,
            "belgian": 0.318,
            "german": 0.327,
        }
        sport_rates = {
            "football": 0.466,
            "rugby": 0.600,
            "swimming": 0.176,
            "tennis": 0.485,
            "chess": 0.018,
            "cricket": 0.118,
        }
        grade_rates = {
            "50s": 0.452,
            "60s": 0.271,
            "70s": 0.232,
        }
        debate_rates = {True: 0.797, False: 0.159}
        programming_rates = {True: 0.163, False: 0.385}
        international_rates = {True: 0.284, False: 0.327}
        entrepeneur_rates = {True: 0.775, False: 0.163}
        languages_rates = {
            0: 0.0,
            1: 0.021,
            2: 0.626,
            3: 0.946,
        }
        exact_study_rates = {True: 0.153, False: 0.461}
        degree_rates = {
            "bachelor": 0.272,
            "master": 0.372,
            "phd": 0.294,
        }

        # --- feature weights ---
        # Increase weight for fields that had very strong signals in the data:
        # debateclub, entrepeneur_exp, and languages.
        weights = {
            "base": 1,
            "gender": 1,
            "age": 1,
            "nationality": 1,
            "sport": 1,
            "university_grade": 1,
            "debateclub": 3,
            "programming_exp": 1,
            "international_exp": 1,
            "entrepeneur_exp": 3,
            "languages": 3,
            "exact_study": 1,
            "degree": 1,
        }

        weighted_sum = 0.0
        total_weight = 0.0

        # include base
        weighted_sum += base_rate * weights["base"]
        total_weight += weights["base"]

        # gender
        gender = inputs.get("gender")
        if gender is not None:
            gkey = str(gender).strip().lower()
            if gkey in gender_rates:
                w = weights["gender"]
                weighted_sum += gender_rates[gkey] * w
                total_weight += w

        # age -> bucket
        age_val = safe_int(inputs.get("age"), None)
        if age_val is not None:
            if age_val <= 25:
                r = age_rates["le_25"]
            else:
                r = age_rates["gt_25"]
            w = weights["age"]
            weighted_sum += r * w
            total_weight += w

        # nationality
        nat = inputs.get("nationality")
        if nat is not None:
            nkey = str(nat).strip().lower()
            if nkey in nationality_rates:
                w = weights["nationality"]
                weighted_sum += nationality_rates[nkey] * w
                total_weight += w

        # sport
        sport = inputs.get("sport")
        if sport is not None:
            skey = str(sport).strip().lower()
            if skey in sport_rates:
                w = weights["sport"]
                weighted_sum += sport_rates[skey] * w
                total_weight += w

        # university_grade -> 50s/60s/70s if applicable
        ug = safe_int(inputs.get("university_grade"), None)
        if ug is not None:
            if 50 <= ug <= 59:
                r = grade_rates["50s"]
            elif 60 <= ug <= 69:
                r = grade_rates["60s"]
            elif 70 <= ug <= 79:
                r = grade_rates["70s"]
            else:
                r = None
            if r is not None:
                w = weights["university_grade"]
                weighted_sum += r * w
                total_weight += w

        # debateclub
        debate = parse_bool(inputs.get("debateclub"))
        w = weights["debateclub"]
        weighted_sum += debate_rates[debate] * w
        total_weight += w

        # programming_exp
        prog = parse_bool(inputs.get("programming_exp"))
        w = weights["programming_exp"]
        weighted_sum += programming_rates[prog] * w
        total_weight += w

        # international_exp
        intl = parse_bool(inputs.get("international_exp"))
        w = weights["international_exp"]
        weighted_sum += international_rates[intl] * w
        total_weight += w

        # entrepeneur_exp
        entrep = parse_bool(inputs.get("entrepeneur_exp"))
        w = weights["entrepeneur_exp"]
        weighted_sum += entrepeneur_rates[entrep] * w
        total_weight += w

        # languages (number)
        lang_num = safe_int(inputs.get("languages"), None)
        if lang_num is not None and lang_num in languages_rates:
            w = weights["languages"]
            weighted_sum += languages_rates[lang_num] * w
            total_weight += w
        # if unseen language count, skip it

        # exact_study
        exact = parse_bool(inputs.get("exact_study"))
        w = weights["exact_study"]
        weighted_sum += exact_study_rates[exact] * w
        total_weight += w

        # degree
        deg = inputs.get("degree")
        if deg is not None:
            dkey = str(deg).strip().lower()
            if dkey in degree_rates:
                w = weights["degree"]
                weighted_sum += degree_rates[dkey] * w
                total_weight += w

        # Final weighted average and decision
        # total_weight should always be > 0 because base is always included
        avg_prob = weighted_sum / total_weight if total_weight > 0 else base_rate

        decision = "yes" if avg_prob >= base_rate else "no"

        return dspy.Prediction(decision=decision)

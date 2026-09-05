class ScoNeSignatureModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # Single LM step: given a premise (context) and a wrapped question asking
        # "Can we logically conclude for sure that <hypothesis>?", produce exactly
        # one token: "Yes" or "No".
        #
        # Instructions encode the intended logical/lexical rules so the LM doesn't
        # have to guess task conventions:
        #
        # - Treat `context` as the only premises. Answer "Yes" iff the hypothesis
        #   (the content asked about in `question`) is a logical consequence of
        #   the context under ordinary lexical/hypernym meaning. Answer "No"
        #   otherwise.
        #
        # - Hypernym rule: if the context asserts a specific category/item and the
        #   hypothesis replaces it with a strict generalization (e.g. "papayas"
        #   -> "produce", "bassoon" -> "instrument", "sombrero" -> "hat",
        #   "flowers" -> "plants", "pizza" -> "food"), output Yes.
        #   Do NOT infer the reverse: general -> specific is No.
        #
        # - Synonym/subtype rule: if hypothesis is a valid synonym or equivalent
        #   label for the same thing in the context, output Yes. If the relation
        #   is uncertain or context doesn't guarantee it, output No.
        #
        # - Quantifiers and existence: respect quantifiers ("a", "there is",
        #   "a single", "two people"). Do not generalize from "some X" to
        #   "all Y". Example: "plays an instrument" from "plays a piano" => Yes.
        #   "plays a piano" from "plays an instrument" => No.
        #
        # - Negation and scope: handle polarity carefully. "It is false that the
        #   boy does not play an instrument" entails the boy plays some instrument
        #   (so instrument-level generalizations are Yes) but does not entail a
        #   particular instrument (No). Double negation and "not the case that
        #   ... not ..." should be interpreted logically.
        #
        # - Do not assume unstated facts (locations, attributes, counts) or world
        #   knowledge beyond common lexical hypernymy/synonymy. If the hypothesis
        #   adds information not in the context (e.g. a different subtype,
        #   extra attribute), answer No.
        #
        # Output requirement: reply with exactly one word, either "Yes" or "No",
        # capitalized and nothing else.
        self.classify = dspy.Predict(dspy.Signature(
            "context: str, question: str -> answer: str",
            "You are given a premise `context` and a `question` that asks: "
            '"Can we logically conclude for sure that <hypothesis>?". '
            "Determine whether the hypothesis necessarily follows from the context "
            "under ordinary meanings and lexical relations. Follow these rules:\n"
            "1) If the hypothesis is a strict generalization/hypernym of something "
            "explicitly stated in the context (e.g. 'papayas' -> 'produce', "
            "'piano' -> 'instrument', 'sombrero' -> 'hat', 'flowers' -> 'plants', "
            "'pizza' -> 'food'), answer Yes. The reverse (general->specific) is No.\n"
            "2) If the hypothesis is a clear synonym or equivalent label of what's "
            "stated, answer Yes. If equivalence is not guaranteed, answer No.\n"
            "3) Respect quantifiers and negation. Existential claims in the context "
            "entail existential hypernyms but not specific subtypes. Double negation "
            "and 'not the case that ... not ...' follow standard logical rules.\n"
            "4) Do NOT assume any unstated facts or additional properties. If the "
            "hypothesis adds information not present or could be false given the "
            "context, answer No.\n"
            "Return exactly one token: either 'Yes' or 'No' (capitalized), and "
            "nothing else."
        ))

    def forward(self, **inputs):
        # Call the LM predictor and coerce its output to exactly "Yes" or "No".
        pred = self.classify(context=inputs["context"], question=inputs["question"])
        raw = ""
        # Unwrap returned field robustly (the predictor declared 'answer')
        try:
            raw = "" if pred.answer is None else str(pred.answer)
        except Exception:
            raw = str(pred)
        out = raw.strip().splitlines()[0].strip()
        out_low = out.lower()
        if out_low.startswith("y"):
            final = "Yes"
        elif out_low.startswith("n"):
            final = "No"
        else:
            # Fallback deterministic heuristic when LM output not clearly yes/no:
            # Conservative default: if hypothesis contains a broader term that is
            # lexical hypernym of a word in context, answer Yes; otherwise No.
            # Simple substring-based hypernym map derived from common patterns.
            hypo = inputs["question"].lower()
            ctx = inputs["context"].lower()
            # try to extract the hypothesis phrase by removing the common wrapper
            # "can we logically conclude for sure that" if present
            import re
            m = re.search(r"conclude for sure that (.*)\?*$", hypo)
            if m:
                hypo_phrase = m.group(1)
            else:
                # fall back to whole question text
                hypo_phrase = hypo
            # small hypernym map capturing frequent dataset relations
            hypernym_map = {
                "produce": ["papaya", "papayas", "celery", "papayas,", "papayas."],
                "instrument": ["piano", "violin", "guitar", "bassoon", "xylophone", "saxophone",
                               "banjo", "clarinet", "sousaphone", "horn", "bongo", "organ",
                               "drum", "ukulele", "triangle", "cornet", "mandolin", "fiddle",
                               "sit ar", "sitar", "piccolo", "saxophone"],
                "boat": ["gondola", "canoe", "houseboat", "barge", "motorboat", "tugboat", "boat", "ship"],
                "person": ["musher", "slaver", "official", "skier", "stagehand", "psychic",
                           "matriarch", "waiter", "granny", "mother", "widow", "lady", "mistress",
                           "bridesmaid", "girl", "boy", "man", "woman", "gent", "gentleman"],
                "food": ["pizza", "curry", "falafel", "naan", "pastry", "spaghetti", "pretzel",
                         "sandwich", "burrito", "tofu", "oatmeal", "burrito", "sandwich", "pastry",
                         "pretzel", "pizza", "curry", "bento"],
                "plant": ["flower", "flowers", "lily", "lilies", "plant", "plants", "reeds"],
                "fish": ["marlin", "marlins", "tuna", "herrings", "herring", "catfish", "fish"],
                "hat": ["sombrero", "hat"],
                "gun": ["pistol", "gun", "rifle", "twenty-two", "rifle", "shotgun"],
                "tree": ["ash", "willow", "birch", "maple", "fir", "cedar", "elm", "magnolia", "pine", "tree"]
            }
            # check if any hypernym word occurs in hypothesis and a hyponym variant occurs in context
            found_yes = False
            for hyper, hypos in hypernym_map.items():
                if hyper in hypo_phrase:
                    for h in hypos:
                        if h in ctx:
                            found_yes = True
                            break
                # also the reverse: hypo specific in hypo_phrase and hyper in context -> No (don't set found_yes)
                if found_yes:
                    break
            final = "Yes" if found_yes else "No"
        return dspy.Prediction(answer=final)

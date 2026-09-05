class SolveModule(dspy.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **inputs):
        def to_float(x):
            if x is None:
                return None
            try:
                return float(x.strip())
            except Exception:
                try:
                    return float(x)
                except Exception:
                    return None

        pl = to_float(inputs.get("petal_length"))
        pw = to_float(inputs.get("petal_width"))
        sl = to_float(inputs.get("sepal_length"))
        sw = to_float(inputs.get("sepal_width"))

        # Primary rule: petal length is the most discriminative feature for iris species.
        if pl is not None:
            # Clear separation: very short petals -> setosa, medium -> versicolor, long -> virginica
            if pl < 2.5:
                answer = "setosa"
            elif pl < 5.0:
                answer = "versicolor"
            else:
                answer = "virginica"
        elif pw is not None:
            # Fallback to petal width if petal length missing
            if pw < 0.8:
                answer = "setosa"
            elif pw < 1.8:
                answer = "versicolor"
            else:
                answer = "virginica"
        else:
            # Final fallback: use sepal area heuristic if petal measurements unavailable
            if sl is None or sw is None:
                # If nothing parseable, default to the middle class
                answer = "versicolor"
            else:
                area = sl * sw
                if area < 16.0:
                    answer = "setosa"
                elif area < 20.0:
                    answer = "versicolor"
                else:
                    answer = "virginica"

        return dspy.Prediction(answer=answer)

class SolveModule(dspy.Module):
    """
    Given the petal and sepal dimensions in cm, predict the iris species.
    Input fields (strings): petal_length, petal_width, sepal_length, sepal_width
    Output: answer (str) -- one of: setosa, versicolor, virginica
    """
    def __init__(self):
        super().__init__()

    def forward(self, **inputs):
        # Helper: robustly parse a numeric string into a float
        def to_float(s, name):
            if s is None:
                raise ValueError(f"missing input '{name}'")
            if not isinstance(s, str):
                # attempt to coerce non-strings
                try:
                    return float(s)
                except Exception:
                    raise ValueError(f"could not parse '{name}' as float")
            s2 = s.strip()
            # accept comma as decimal separator (common in some datasets)
            s2 = s2.replace(",", ".")
            if s2 == "":
                raise ValueError(f"empty input for '{name}'")
            try:
                return float(s2)
            except Exception as e:
                raise ValueError(f"could not parse '{name}' as float: {e}")

        petal_length = to_float(inputs.get("petal_length"), "petal_length")
        petal_width  = to_float(inputs.get("petal_width"),  "petal_width")
        # sepal measurements are not required for this deterministic rule,
        # but parse them to validate the inputs if provided
        _ = to_float(inputs.get("sepal_length"), "sepal_length")
        _ = to_float(inputs.get("sepal_width"), "sepal_width")

        # Deterministic rule derived from standard Iris separations:
        #  - setosa is separable by small petal length
        #  - between versicolor and virginica petal width is a good separator
        # These thresholds are commonly used in simple decision trees.
        if petal_length <= 2.45:
            label = "setosa"
        else:
            if petal_width <= 1.75:
                label = "versicolor"
            else:
                label = "virginica"

        return dspy.Prediction(answer=label)

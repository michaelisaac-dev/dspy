class SigModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(dspy.Signature({
    "petal_length": (str, dspy.InputField()),
    "petal_width": (str, dspy.InputField()),
    "sepal_length": (str, dspy.InputField()),
    "sepal_width": (str, dspy.InputField()),
    "answer": (str, dspy.OutputField(desc="setosa, versicolor, or virginica")),
}, "Given the petal and sepal dimensions in cm, predict the iris species."))

    def forward(self, **inputs):
        result = self.predict(**inputs)
        return dspy.Prediction(answer=result.answer)

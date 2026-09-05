"""LangProBe x Flex pilot: HeartDisease benchmark.

Uses LangProBe's own benchmark class (data, seeded splits) and metric
(answer_exact_match) with the local dspy checkout (3.3.x, Flex).

Arms:
  B0a Predict (unoptimized)          B0b CoT (unoptimized)
  B0c Flex identity (unoptimized)    B1  GEPA on CoT   (prompt space)
  B2  GEPA on Flex  (code space)     -- same budget, same data, same metric.

Run:  run_pilot.py smoke | baselines | gepa_cot | gepa_flex | report
"""

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "langprobe"))

# repo .env wins over any stale shell exports
for line in (HERE.parent.parent.parent / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ[k.strip()] = v.strip().strip('"').strip("'")

from langProBe.HeartDisease.HeartDisease_data import HeartDiseaseBench
from langProBe.HeartDisease.HeartDisease_program import HeartDiseaseSignature

import dspy

TASK_LM = "openai/gpt-4o-mini"
REFLECTION_LM = "openai/gpt-5-mini"
N_TEST = None  # full LangProBe "lite" test split (~151)
VALSET_N = 40
MAX_METRIC_CALLS = 500
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def em_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    return float(dspy.evaluate.answer_exact_match(gold, pred, trace))


class FlexHeart(dspy.Module):
    def __init__(self):
        super().__init__()
        self.flex = dspy.Flex(HeartDiseaseSignature)

    def forward(self, **kwargs):
        return self.flex(**kwargs)


def build_programs():
    return {
        "predict_base": dspy.Predict(HeartDiseaseSignature),
        "cot_base": dspy.ChainOfThought(HeartDiseaseSignature),
        "flex_base": FlexHeart(),
    }


def evaluate(program, testset, task_lm, tag):
    before = len(task_lm.history)
    t0 = time.time()
    ev = dspy.Evaluate(devset=testset, metric=em_metric, num_threads=8,
                       display_progress=True, provide_traceback=True)
    result = ev(program)
    score = result.score if hasattr(result, "score") else float(result)
    calls = len(task_lm.history) - before
    cost = sum((h.get("cost") or 0) for h in task_lm.history[before:])
    row = {"arm": tag, "score": score, "task_lm_calls": calls,
           "task_lm_cost_usd": round(cost, 4), "n_test": len(testset),
           "wall_s": round(time.time() - t0, 1)}
    print(f"RESULT {json.dumps(row)}")
    path = OUT / "results.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def gepa(reflection_lm):
    return dspy.GEPA(metric=em_metric, max_metric_calls=MAX_METRIC_CALLS,
                     reflection_lm=reflection_lm, num_threads=8, seed=0,
                     track_stats=True)


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    task_lm = dspy.LM(TASK_LM, max_tokens=1000, temperature=0.0, cache=True)
    reflection_lm = dspy.LM(REFLECTION_LM, temperature=1.0, max_tokens=20000)
    dspy.configure(lm=task_lm)

    bench = HeartDiseaseBench(dataset_mode="lite")
    testset = bench.test_set if N_TEST is None else bench.test_set[:N_TEST]
    trainset = bench.train_set
    valset = bench.val_set[:VALSET_N]
    print(f"splits: train={len(trainset)} val={len(valset)} test={len(testset)}")

    if phase == "smoke":
        progs = build_programs()
        smoke = testset[:5]
        for tag, p in progs.items():
            evaluate(p, smoke, task_lm, f"smoke_{tag}")
        return

    if phase == "baselines":
        progs = build_programs()
        for tag, p in progs.items():
            evaluate(p, testset, task_lm, tag)
        return

    if phase == "gepa_cot":
        student = dspy.ChainOfThought(HeartDiseaseSignature)
        compiled = gepa(reflection_lm).compile(student, trainset=trainset, valset=valset)
        compiled.save(OUT / "gepa_cot.json")
        evaluate(compiled, testset, task_lm, "gepa_cot")
        return

    if phase == "gepa_flex":
        student = FlexHeart()
        compiled = gepa(reflection_lm).compile(student, trainset=trainset, valset=valset)
        compiled.save(OUT / "gepa_flex.json")
        src = compiled.flex.module_src
        (OUT / "gepa_flex_module_src.py").write_text(src or "<none>")
        print(f"module_src saved ({len(src or '')} chars)")
        evaluate(compiled, testset, task_lm, "gepa_flex")
        return

    print(f"unknown phase {phase}")


if __name__ == "__main__":
    main()

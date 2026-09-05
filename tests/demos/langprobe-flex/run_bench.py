"""LangProBe x Flex: generalized runner (HeartDisease / Iris / Scone).

Arms per benchmark, all on LangProBe's own splits + metric (answer_exact_match):
  baselines  - Predict, CoT, Flex-identity (unoptimized)
  gepa_cot   - GEPA over prompt space (CoT student), 500 metric calls
  gepa_flex  - GEPA over code space (Flex student), 500 metric calls
  oneshot    - B3 attribution arm: reflection LM writes module_src once (best-of-3
               on the valset, no optimization loop) -> is the win codegen or optimization?

Usage: run_bench.py <heart|iris|scone> <baselines|gepa_cot|gepa_flex|oneshot> [seed]
"""

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "langprobe"))

for line in (HERE.parent.parent.parent / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ[k.strip()] = v.strip().strip('"').strip("'")

import dspy
from dspy.predict.flex.primitives_doc import PRIMITIVES_CATALOG

TASK_LM = "openai/gpt-4o-mini"
REFLECTION_LM = "openai/gpt-5-mini"
VALSET_N = 40
MAX_METRIC_CALLS = 500
ONESHOT_K = 3
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def registry(bench_name):
    if bench_name == "heart":
        from langProBe.HeartDisease.HeartDisease_data import HeartDiseaseBench
        from langProBe.HeartDisease.HeartDisease_program import HeartDiseaseSignature
        return HeartDiseaseBench, HeartDiseaseSignature
    if bench_name == "iris":
        from langProBe.Iris.Iris_data import IrisBench
        from langProBe.Iris.Iris_program import Sig
        return IrisBench, Sig
    if bench_name == "scone":
        from langProBe.scone.scone_data import SconeBench
        from langProBe.scone.scone_program import ScoNeSignature
        return SconeBench, ScoNeSignature
    if bench_name == "iris_fixed":
        # LangProBe's Iris split never shuffles the class-sorted source file: train is all
        # setosa, test is only virginica/versicolor. This variant reshuffles (seeded) before
        # splitting, same sizes — NOT comparable to the paper's Iris numbers.
        import random

        from langProBe.Iris.Iris_data import IrisBench
        from langProBe.Iris.Iris_program import Sig

        class IrisFixedBench(IrisBench):
            def init_dataset(self):
                super().init_dataset()
                full = self.dataset + self.test_set
                random.Random(42).shuffle(full)
                self.test_set = full[len(full) // 2:]
                self.dataset = full[: len(full) // 2]
                self.train_set = self.dataset[:15]
                self.val_set = self.dataset[15:]
                self.dev_set = self.dataset

        return IrisFixedBench, Sig
    raise SystemExit(f"unknown bench {bench_name}")


def em_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    return float(dspy.evaluate.answer_exact_match(gold, pred, trace))


def make_flex_program(signature):
    class FlexProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.flex = dspy.Flex(signature)

        def forward(self, **kwargs):
            return self.flex(**kwargs)

    return FlexProgram()


def evaluate(program, testset, task_lm, tag, bench, seed):
    before = len(task_lm.history)
    t0 = time.time()
    ev = dspy.Evaluate(devset=testset, metric=em_metric, num_threads=8,
                       display_progress=True, provide_traceback=True)
    result = ev(program)
    score = result.score if hasattr(result, "score") else float(result)
    calls = len(task_lm.history) - before
    cost = sum((h.get("cost") or 0) for h in task_lm.history[before:])
    row = {"bench": bench, "seed": seed, "arm": tag, "score": score,
           "task_lm_calls": calls, "task_lm_cost_usd": round(cost, 4),
           "n_test": len(testset), "wall_s": round(time.time() - t0, 1)}
    print(f"RESULT {json.dumps(row)}")
    with open(OUT / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def render_train_examples(trainset, signature):
    lines = []
    for ex in trainset:
        ins = {k: ex[k] for k in ex.inputs().keys()}
        outs = {k: ex[k] for k in signature.output_fields}
        lines.append(f"inputs={json.dumps(ins)}  ->  expected={json.dumps(outs)}")
    return "\n".join(lines)


def oneshot_candidates(flex, trainset, reflection_lm, k, seed):
    spec = flex._flex_ctx.render_signature_spec()
    prompt = (
        "Write the full source of a single `dspy.Module` subclass implementing this task. "
        "It runs in a sandboxed interpreter with the primitives below. Favor deterministic "
        "code where the data allows it; you may construct and call dspy predictors when "
        "genuinely needed. Return ONLY the class source, no fences, no prose.\n\n"
        f"{spec}\n\nTraining examples:\n{render_train_examples(trainset, flex.signature)}\n\n"
        f"{PRIMITIVES_CATALOG}\n\n"
        f"Reference: the current implementation is:\n{flex.module_src}\n"
    )
    outs = []
    for i in range(k):
        # nonce defeats the LM cache so each candidate is an independent sample
        nonced = prompt + f"\n(sample id: {seed}.{i})\n"
        with dspy.context(lm=reflection_lm):
            text = reflection_lm(messages=[{"role": "user", "content": nonced}])[0]
        src = text.strip()
        if src.startswith("```"):
            src = src.strip("`\n")
            src = src.partition("\n")[2] if src.startswith("python") else src
        outs.append(src)
    return outs


def main():
    bench_name, phase = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    BenchCls, Sig = registry(bench_name)

    task_lm = dspy.LM(TASK_LM, max_tokens=1000, temperature=0.0, cache=True)
    reflection_lm = dspy.LM(REFLECTION_LM, temperature=1.0, max_tokens=20000)
    dspy.configure(lm=task_lm)

    bench = BenchCls(dataset_mode="lite")
    testset, trainset, valset = bench.test_set, bench.train_set, bench.val_set[:VALSET_N]
    print(f"{bench_name}: train={len(trainset)} val={len(valset)} test={len(testset)} seed={seed}")

    if phase == "baselines":
        evaluate(dspy.Predict(Sig), testset, task_lm, "predict_base", bench_name, seed)
        evaluate(dspy.ChainOfThought(Sig), testset, task_lm, "cot_base", bench_name, seed)
        evaluate(make_flex_program(Sig), testset, task_lm, "flex_base", bench_name, seed)
        return

    if phase in ("gepa_cot", "gepa_flex"):
        student = dspy.ChainOfThought(Sig) if phase == "gepa_cot" else make_flex_program(Sig)
        opt = dspy.GEPA(metric=em_metric, max_metric_calls=MAX_METRIC_CALLS,
                        reflection_lm=reflection_lm, num_threads=8, seed=seed,
                        track_stats=True)
        compiled = opt.compile(student, trainset=trainset, valset=valset)
        compiled.save(OUT / f"{bench_name}_{phase}_s{seed}.json")
        if phase == "gepa_flex":
            src = compiled.flex.module_src or "<none>"
            (OUT / f"{bench_name}_{phase}_s{seed}_module_src.py").write_text(src)
        evaluate(compiled, testset, task_lm, phase, bench_name, seed)
        return

    if phase == "gepa_flex_lam":
        lam = float(sys.argv[4])

        def lam_metric(gold, pred, trace=None, pred_name=None, pred_trace=None,
                       program_trace=None):
            em = float(dspy.evaluate.answer_exact_match(gold, pred, trace))
            calls = len(program_trace or [])
            fb = (f"Answer {'correct' if em else 'WRONG'}; used {calls} LM call(s). "
                  f"Each LM call costs {lam} of score — settle what you can in code, "
                  f"but never at the price of correctness.")
            return dspy.Prediction(score=em - lam * calls, feedback=fb)

        student = make_flex_program(Sig)
        tag = f"gepa_flex_lam{lam:g}"
        if len(sys.argv) > 5 and sys.argv[5] == "warm":
            # continual recompilation: start from the lambda=0 artifact and let the
            # penalty trim cost without rediscovering the instruction
            student.load(OUT / f"{bench_name}_gepa_flex_s{seed}.json")
            tag = f"gepa_flex_lam{lam:g}_warm"
        opt = dspy.GEPA(metric=lam_metric, max_metric_calls=MAX_METRIC_CALLS,
                        reflection_lm=reflection_lm, num_threads=8, seed=seed,
                        track_stats=True)
        compiled = opt.compile(student, trainset=trainset, valset=valset)
        compiled.save(OUT / f"{bench_name}_{tag}_s{seed}.json")
        (OUT / f"{bench_name}_{tag}_s{seed}_module_src.py").write_text(
            compiled.flex.module_src or "<none>")
        evaluate(compiled, testset, task_lm, tag, bench_name, seed)
        return

    if phase == "oneshot":
        prog = make_flex_program(Sig)
        cands = oneshot_candidates(prog.flex, trainset, reflection_lm, ONESHOT_K, seed)
        best_src, best_score = None, -1.0
        for i, src in enumerate(cands):
            try:
                prog.flex._bind_code(src)
                ev = dspy.Evaluate(devset=valset, metric=em_metric, num_threads=8,
                                   display_progress=False)
                s = ev(prog)
                s = s.score if hasattr(s, "score") else float(s)
            except Exception as e:
                print(f"candidate {i} failed: {type(e).__name__}: {e}")
                s = -1.0
            print(f"oneshot candidate {i}: valset {s}")
            if s > best_score:
                best_src, best_score = src, s
        prog.flex._bind_code(best_src)
        (OUT / f"{bench_name}_oneshot_s{seed}_module_src.py").write_text(best_src)
        evaluate(prog, testset, task_lm, "oneshot_codegen", bench_name, seed)
        return

    raise SystemExit(f"unknown phase {phase}")


if __name__ == "__main__":
    main()

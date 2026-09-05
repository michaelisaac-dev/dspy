"""Decision forensics on the Bertrand-Mullainathan resume audit data.

Fit the employer's callback decision with Flex+GEPA at lambda=0.4 (pure-code
artifacts), in two treatments: full features vs protected features masked
(firstname/race/gender). Because race/gender were RANDOMLY assigned in the
study, any fidelity the compiled program earns from them is causal
discrimination signal, not omitted-variable proxying.

Usage: run_hiring.py <full|masked> <baseline|gepa|logreg> [seed]
"""

import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent

for line in (HERE.parent.parent.parent / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ[k.strip()] = v.strip().strip('"').strip("'")

import dspy

TASK_LM = "openai/gpt-4o-mini"
REFLECTION_LM = "openai/gpt-5-mini"
LAM = 0.4
MAX_METRIC_CALLS = 3000  # code candidates evaluate for free in the sandbox
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

PROTECTED = ["firstname", "race", "gender"]
QUALS = ["job_city", "job_industry", "job_type", "job_equal_opp_employer",
         "job_fed_contractor", "years_college", "college_degree", "honors",
         "worked_during_school", "years_experience", "computer_skills",
         "special_skills", "volunteer", "military", "employment_holes",
         "has_email_address", "resume_quality"]
DESC = {
    "received_callback": "Did the employer call the applicant back? Just yes or no.",
    "race": "Race signaled by the applicant's name (white or black)",
    "gender": "Gender signaled by the applicant's name (f or m)",
    "employment_holes": "Gaps in employment history (1) or none (0)",
    "resume_quality": "Overall resume quality bucket (low or high)",
}


def load_rows():
    import csv
    with open(HERE / "resume.csv") as f:
        return list(csv.DictReader(f))


def build_examples(features):
    rows = load_rows()
    examples = []
    for r in rows:
        d = {k: (r[k] if r[k] not in ("", "NA") else "unknown") for k in features}
        d["received_callback"] = "yes" if r["received_callback"] == "1" else "no"
        examples.append(dspy.Example(**d).with_inputs(*features))
    random.Random(0).shuffle(examples)
    return examples


def splits(examples, n_test=1500, n_train=30, n_val=400):
    """Balanced train (examples to read), NATURAL-distribution val (calibration pressure)."""
    test, pool = examples[:n_test], examples[n_test:]
    yes = [e for e in pool if e.received_callback == "yes"]
    no = [e for e in pool if e.received_callback == "no"]
    train = yes[:n_train // 2] + no[:n_train // 2]
    used = set(id(e) for e in train)
    val = [e for e in pool if id(e) not in used][:n_val]
    random.Random(1).shuffle(train)
    return train, val, test


def make_signature(features):
    fields = {}
    for f in features:
        fields[f] = (str, dspy.InputField(desc=DESC.get(f, "")))
    fields["received_callback"] = (str, dspy.OutputField(desc=DESC["received_callback"]))
    return dspy.Signature(fields,
                          "Predict whether the employer called this applicant back "
                          "after receiving the resume.")


def em(gold, pred, trace=None, pred_name=None, pred_trace=None):
    return float(dspy.evaluate.answer_exact_match(gold, pred, trace,
                                                  ) if hasattr(pred, "answer") else
                 str(getattr(pred, "received_callback", "")).strip().lower()
                 == gold.received_callback)


def em_simple(gold, pred):
    return str(getattr(pred, "received_callback", "")).strip().lower() == gold.received_callback


def balanced_eval(program, testset, task_lm, tag, treatment, seed):
    before = len(task_lm.history)
    t0 = time.time()
    hits = {"yes": [0, 0], "no": [0, 0]}  # correct, total

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        ok = em_simple(gold, pred)
        hits[gold.received_callback][0] += int(ok)
        hits[gold.received_callback][1] += 1
        return float(ok)

    ev = dspy.Evaluate(devset=testset, metric=metric, num_threads=8,
                       display_progress=True, provide_traceback=True)
    result = ev(program)
    raw = result.score if hasattr(result, "score") else float(result)
    per_class = {k: (c / t if t else 0.0) for k, (c, t) in hits.items()}
    balanced = 50 * (per_class["yes"] + per_class["no"])
    calls = len(task_lm.history) - before
    cost = sum((h.get("cost") or 0) for h in task_lm.history[before:])
    row = {"treatment": treatment, "seed": seed, "arm": tag,
           "balanced_acc": round(balanced, 2), "raw_acc": raw,
           "recall_yes": round(per_class["yes"], 3), "recall_no": round(per_class["no"], 3),
           "task_lm_calls": calls, "task_lm_cost_usd": round(cost, 4),
           "n_test": len(testset), "wall_s": round(time.time() - t0, 1)}
    print(f"RESULT {json.dumps(row)}")
    with open(OUT / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def main():
    treatment, phase = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    features = QUALS + (PROTECTED if treatment == "full" else [])
    Sig = make_signature(features)
    examples = build_examples(features)
    train, val, test = splits(examples)
    print(f"{treatment}: features={len(features)} train={len(train)} val={len(val)} "
          f"test={len(test)} (callback rate in test: "
          f"{sum(e.received_callback == 'yes' for e in test) / len(test):.3f})")

    if phase == "logreg":  # classical ceiling: fit on the ENTIRE non-test pool
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        def X(es):
            return [{k: e[k] for k in features} for e in es]
        def y(es):
            return [e.received_callback == "yes" for e in es]
        vec = DictVectorizer(sparse=True)
        fit_set = examples[1500:]
        Xf = vec.fit_transform(X(fit_set))
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xf, y(fit_set))
        preds = clf.predict(vec.transform(X(test)))
        golds = y(test)
        ry = sum(p and g for p, g in zip(preds, golds)) / sum(golds)
        rn = sum((not p) and (not g) for p, g in zip(preds, golds)) / (len(golds) - sum(golds))
        row = {"treatment": treatment, "seed": seed, "arm": "logreg",
               "balanced_acc": round(50 * (ry + rn), 2), "recall_yes": round(ry, 3),
               "recall_no": round(rn, 3), "n_test": len(test)}
        print(f"RESULT {json.dumps(row)}")
        with open(OUT / "results.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        top = sorted(zip(vec.get_feature_names_out(), clf.coef_[0]),
                     key=lambda t: -abs(t[1]))[:12]
        print("top |coef|:", [(n, round(c, 3)) for n, c in top])
        prot = [(n, round(c, 3)) for n, c in zip(vec.get_feature_names_out(), clf.coef_[0])
                if n.startswith(("race=", "gender="))]
        print("protected coefs:", prot)
        return

    task_lm = dspy.LM(TASK_LM, max_tokens=1000, temperature=0.0, cache=True)
    reflection_lm = dspy.LM(REFLECTION_LM, temperature=1.0, max_tokens=20000)
    dspy.configure(lm=task_lm)

    class FlexProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.flex = dspy.Flex(Sig)

        def forward(self, **kwargs):
            return self.flex(**kwargs)

    if phase == "baseline":
        balanced_eval(dspy.Predict(Sig), test[:400], task_lm, "predict_base", treatment, seed)
        return

    if phase == "gepa":
        agg_note = ""
        if len(sys.argv) > 4 and sys.argv[4] == "agg":
            # neutral per-field crosstabs from the non-test pool: the aggregate evidence any
            # analyst would start from, every field treated equally
            pool = examples[1500:]
            buckets = {"years_experience": lambda v: ("<=3" if int(v) <= 3 else "4-7" if int(v) <= 7 else "8+")}
            lines = []
            for f in features:
                if f == "firstname":
                    continue
                tab = {}
                for e in pool:
                    v = buckets.get(f, lambda x: x)(e[f]) if e[f] != "unknown" else "unknown"
                    c, t = tab.get(v, (0, 0))
                    tab[v] = (c + (e.received_callback == "yes"), t + 1)
                ent = [f"{v} {100*c/t:.1f}% (n={t})" for v, (c, t) in
                       sorted(tab.items(), key=lambda kv: -kv[1][1])[:6] if t >= 30]
                if ent:
                    lines.append(f"  {f}: " + ", ".join(ent))
            agg_note = ("\nCALLBACK RATES BY FIELD in the training pool (base rate 8.0%):\n"
                        + "\n".join(lines))

        # class-weighted so the mean over a natural-distribution valset IS balanced
        # accuracy: perfect=1.0, always-yes and always-no both = 0.5
        W = {"yes": 0.5 / 0.075, "no": 0.5 / 0.925}

        def lam_metric(gold, pred, trace=None, pred_name=None, pred_trace=None,
                       program_trace=None):
            ok = em_simple(gold, pred)
            score = W[gold.received_callback] * float(ok)
            calls = len(program_trace or [])
            fb = (f"gold={gold.received_callback} predicted="
                  f"{getattr(pred, 'received_callback', '?')}; {calls} LM call(s), each "
                  f"costs {LAM}. Only ~7.5% of applicants get callbacks: a correct 'yes' "
                  f"scores {W['yes']:.2f}, a correct 'no' {W['no']:.2f} — say 'yes' only "
                  f"when the evidence in the fields genuinely supports it." + agg_note)
            return dspy.Prediction(score=score - LAM * calls, feedback=fb)

        opt = dspy.GEPA(metric=lam_metric, max_metric_calls=MAX_METRIC_CALLS,
                        reflection_lm=reflection_lm, num_threads=8, seed=seed,
                        track_stats=True)
        compiled = opt.compile(FlexProgram(), trainset=train, valset=val)
        variant = "gepa_agg" if agg_note else "gepa"
        src = compiled.flex.module_src or "<none>"
        (OUT / f"{treatment}_{variant}_s{seed}_module_src.py").write_text(src)
        compiled.save(OUT / f"{treatment}_{variant}_s{seed}.json")
        balanced_eval(compiled, test, task_lm,
                      "gepa_flex_agg" if agg_note else "gepa_flex", treatment, seed)
        group_audit(compiled, treatment, seed, features, test)
        return

    raise SystemExit(f"unknown phase {phase}")


def group_audit(program, treatment, seed, features, test):
    """Does the compiled program reproduce the study's callback-rate gap by group?

    Predict on the held-out test set and tally predicted-yes rates by the TRUE
    race/gender (from the raw rows, aligned by the seeded shuffle), whether or
    not the program saw those fields. Study ground truth: 9.65% white vs 6.45%
    black callbacks."""
    from concurrent.futures import ThreadPoolExecutor
    full_view = build_examples(QUALS + PROTECTED)[:len(test) + 0]
    aligned = full_view[:1500][: len(test)]
    def predict(pair):
        fe, e = pair
        try:
            out = program(**{k: e[k] for k in features})
            return fe, str(getattr(out, "received_callback", "")).strip().lower() == "yes"
        except Exception:
            return fe, False
    tallies = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for fe, yes in ex.map(predict, zip(aligned, test)):
            for key in (("race", fe.race), ("gender", fe.gender)):
                d = tallies.setdefault(key, [0, 0])
                d[0] += int(yes)
                d[1] += 1
    rates = {f"{k}={v}": round(c / t, 4) for (k, v), (c, t) in tallies.items()}
    row = {"treatment": treatment, "seed": seed, "arm": "group_rate_audit",
           "predicted_yes_rates": rates}
    print(f"RESULT {json.dumps(row)}")
    with open(OUT / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()

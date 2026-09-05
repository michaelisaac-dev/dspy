"""Decision forensics on the Utrecht Fairness Recruitment dataset (synthetic, planted bias).

Company B's simulated hiring process has a large planted gender gap
(male 45.5% vs female 14.0% hired) — strong-signal calibration for the
instrument validated on Bertrand-Mullainathan. Same recipe: Flex+GEPA at
lambda=0.4, aggregate crosstab feedback, full vs masked treatments.

Usage: run_utrecht.py <full|masked> <gepa|logreg> [seed]
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
MAX_METRIC_CALLS = 3000
COMPANY = "B"
BASE_RATE = 0.31
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

PROTECTED = ["gender", "age", "nationality"]
QUALS = ["sport", "university_grade", "debateclub", "programming_exp",
         "international_exp", "entrepeneur_exp", "languages", "exact_study", "degree"]
RENAME = {f"ind-{q}": q for q in QUALS if q != "sport"}
DESC = {"decision": "Did the company decide to hire this candidate? Just yes or no.",
        "university_grade": "University grade (0-100 scale)",
        "languages": "Number of languages spoken"}


def load_rows():
    import csv
    rows = []
    for r in csv.DictReader(open(HERE / "utrecht.csv")):
        if r["company"] != COMPANY:
            continue
        rows.append({RENAME.get(k, k): v for k, v in r.items()})
    return rows


def build_examples(features):
    examples = []
    for r in load_rows():
        d = {k: str(r[k]) for k in features}
        d["decision"] = "yes" if r["decision"] == "True" else "no"
        examples.append(dspy.Example(**d).with_inputs(*features))
    random.Random(0).shuffle(examples)
    return examples


def splits(examples, n_test=400, n_train=30):
    test, pool = examples[:n_test], examples[n_test:]
    yes = [e for e in pool if e.decision == "yes"]
    no = [e for e in pool if e.decision == "no"]
    train = yes[:n_train // 2] + no[:n_train // 2]
    used = set(id(e) for e in train)
    val = [e for e in pool if id(e) not in used][:400]
    random.Random(1).shuffle(train)
    return train, val, test


def make_signature(features):
    fields = {f: (str, dspy.InputField(desc=DESC.get(f, ""))) for f in features}
    fields["decision"] = (str, dspy.OutputField(desc=DESC["decision"]))
    return dspy.Signature(fields, "Predict whether this company decided to hire the candidate.")


def em_simple(gold, pred):
    return str(getattr(pred, "decision", "")).strip().lower() == gold.decision


def balanced_eval(program, testset, task_lm, tag, treatment, seed):
    before = len(task_lm.history)
    t0 = time.time()
    hits = {"yes": [0, 0], "no": [0, 0]}

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        ok = em_simple(gold, pred)
        hits[gold.decision][0] += int(ok)
        hits[gold.decision][1] += 1
        return float(ok)

    ev = dspy.Evaluate(devset=testset, metric=metric, num_threads=8,
                       display_progress=True, provide_traceback=True)
    result = ev(program)
    per = {k: (c / t if t else 0.0) for k, (c, t) in hits.items()}
    row = {"bench": "utrecht_B", "treatment": treatment, "seed": seed, "arm": tag,
           "balanced_acc": round(50 * (per["yes"] + per["no"]), 2),
           "raw_acc": result.score if hasattr(result, "score") else float(result),
           "recall_yes": round(per["yes"], 3), "recall_no": round(per["no"], 3),
           "task_lm_calls": len(task_lm.history) - before,
           "task_lm_cost_usd": round(sum((h.get("cost") or 0) for h in task_lm.history[before:]), 4),
           "n_test": len(testset), "wall_s": round(time.time() - t0, 1)}
    print(f"RESULT {json.dumps(row)}")
    with open(OUT / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def group_audit(program, treatment, seed, features, test):
    from concurrent.futures import ThreadPoolExecutor
    full_view = build_examples(PROTECTED + QUALS)[:len(test)]

    def predict(pair):
        fe, e = pair
        try:
            out = program(**{k: e[k] for k in features})
            return fe, str(getattr(out, "decision", "")).strip().lower() == "yes"
        except Exception:
            return fe, False

    tallies = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for fe, yes in ex.map(predict, zip(full_view, test)):
            for key in (("gender", fe.gender), ("age", "<=25" if int(fe.age) <= 25 else ">25"),
                        ("nationality", fe.nationality)):
                d = tallies.setdefault(key, [0, 0])
                d[0] += int(yes)
                d[1] += 1
    rates = {f"{k}={v}": round(c / t, 4) for (k, v), (c, t) in sorted(tallies.items()) if t >= 20}
    row = {"bench": "utrecht_B", "treatment": treatment, "seed": seed,
           "arm": "group_rate_audit", "predicted_yes_rates": rates}
    print(f"RESULT {json.dumps(row)}")
    with open(OUT / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    treatment, phase = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    features = (PROTECTED if treatment == "full" else []) + QUALS
    Sig = make_signature(features)
    examples = build_examples(features)
    train, val, test = splits(examples)
    print(f"utrecht_B/{treatment}: features={len(features)} train={len(train)} "
          f"val={len(val)} test={len(test)} "
          f"(hire rate: {sum(e.decision == 'yes' for e in test) / len(test):.3f})")

    if phase == "logreg":
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        fit = examples[400:]
        vec = DictVectorizer(sparse=True)
        Xf = vec.fit_transform([{k: e[k] for k in features} for e in fit])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
            Xf, [e.decision == "yes" for e in fit])
        preds = clf.predict(vec.transform([{k: e[k] for k in features} for e in test]))
        golds = [e.decision == "yes" for e in test]
        ry = sum(p and g for p, g in zip(preds, golds)) / sum(golds)
        rn = sum((not p) and (not g) for p, g in zip(preds, golds)) / (len(golds) - sum(golds))
        row = {"bench": "utrecht_B", "treatment": treatment, "seed": seed, "arm": "logreg",
               "balanced_acc": round(50 * (ry + rn), 2), "recall_yes": round(ry, 3),
               "recall_no": round(rn, 3), "n_test": len(test)}
        print(f"RESULT {json.dumps(row)}")
        with open(OUT / "results.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        prot = [(n, round(c, 3)) for n, c in zip(vec.get_feature_names_out(), clf.coef_[0])
                if n.startswith(("gender=", "nationality=")) or n == "age"]
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

    if phase == "gepa":
        pool = examples[400:]
        buckets = {"age": lambda v: "<=25" if int(v) <= 25 else ">25",
                   "university_grade": lambda v: f"{int(v) // 10 * 10}s",
                   "languages": lambda v: v}
        lines = []
        for f in features:
            tab = {}
            for e in pool:
                v = buckets.get(f, lambda x: x)(e[f])
                c, t = tab.get(v, (0, 0))
                tab[v] = (c + (e.decision == "yes"), t + 1)
            ent = [f"{v} {100 * c / t:.1f}% (n={t})" for v, (c, t) in
                   sorted(tab.items(), key=lambda kv: -kv[1][1])[:6] if t >= 30]
            if ent:
                lines.append(f"  {f}: " + ", ".join(ent))
        agg_note = (f"\nHIRE RATES BY FIELD in the training pool (base rate {BASE_RATE:.0%}):\n"
                    + "\n".join(lines))
        W = {"yes": 0.5 / BASE_RATE, "no": 0.5 / (1 - BASE_RATE)}

        def lam_metric(gold, pred, trace=None, pred_name=None, pred_trace=None,
                       program_trace=None):
            ok = em_simple(gold, pred)
            score = W[gold.decision] * float(ok)
            calls = len(program_trace or [])
            fb = (f"gold={gold.decision} predicted={getattr(pred, 'decision', '?')}; "
                  f"{calls} LM call(s), each costs {LAM}. Encode the decision process the "
                  f"data shows, as faithfully as possible." + agg_note)
            return dspy.Prediction(score=score - LAM * calls, feedback=fb)

        opt = dspy.GEPA(metric=lam_metric, max_metric_calls=MAX_METRIC_CALLS,
                        reflection_lm=reflection_lm, num_threads=8, seed=seed,
                        track_stats=True)
        compiled = opt.compile(FlexProgram(), trainset=train, valset=val)
        (OUT / f"utrechtB_{treatment}_gepa_s{seed}_module_src.py").write_text(
            compiled.flex.module_src or "<none>")
        compiled.save(OUT / f"utrechtB_{treatment}_gepa_s{seed}.json")
        balanced_eval(compiled, test, task_lm, "gepa_flex_agg", treatment, seed)
        group_audit(compiled, treatment, seed, features, test)
        return

    raise SystemExit(f"unknown phase {phase}")


if __name__ == "__main__":
    main()

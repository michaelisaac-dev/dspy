"""Did GEPA have enough budget? Parses a sweep log and flags runs that were still improving.

The tell is WHERE in the run the best candidate was found. Found early -> the budget was ample and
the search plateaued. Found in the last quarter -> the search was still climbing when the budget ran
out, and more rollouts would likely have helped.

This matters more here than in the other two demos. One metric call is one container episode, so
`max_metric_calls` is small (120, not 600) and the risk of reporting a starved run as a finding is
correspondingly higher -- exactly the mistake the emails demo made at lambda=0.2, where 200 calls
understated the true result on accuracy, call rate and cost simultaneously.

    python sweep_penalties.py ... 2>&1 | tee sweep.log
    python budget_check.py sweep.log
"""

import re
import sys


def analyse(path: str) -> None:
    txt = open(path, errors="ignore").read().replace("\r", "\n")
    sections = re.split(r"=== penalty ", txt)[1:]
    if not sections:
        print(f"{path}: no '=== penalty ...' sections found -- is this a sweep log?")
        return
    for section in sections:
        lam = section.split()[0]
        iters = [int(m) for m in re.findall(r"Iteration (\d+):", section)]
        if not iters:
            print(f"lambda={lam:<5} no iterations logged (crashed early, or resumed from cache)")
            continue
        last = max(iters)
        accepts = re.findall(r"Iteration (\d+): New program candidate index", section)
        bests = [(int(i), float(v)) for i, v in
                 re.findall(r"Iteration (\d+): Best score on valset: ([0-9.]+)", section)]
        base = re.findall(r"Base program full valset score: ([0-9.]+)", section)
        best_iter, best_val = (max(bests, key=lambda t: (t[1], -t[0])) if bests else (None, None))
        ceiling = best_val is not None and best_val >= 0.999
        late = best_iter is not None and last > 0 and best_iter / last > 0.75
        print(f"lambda={lam:<5} iterations={last:<4} proposals={len(re.findall(r'Proposed new text', section)):<4}"
              f" accepted={len(accepts):<3} base_val={base[0] if base else '?':<6} best_val={best_val}"
              f" @iter {best_iter}")
        if best_val is not None and base and float(base[0]) >= best_val:
            print("        -> the base harness was never beaten. Either the budget was far too small, "
                  "or every proposal failed to bind (check for TRUNCATED calls in the sweep output).")
        elif ceiling:
            print("        -> val CEILING reached; extra budget cannot improve selection.")
        elif late:
            print(f"        -> best found in the final quarter ({best_iter}/{last}): STILL CLIMBING, "
                  f"raise max_metric_calls.")
        else:
            print(f"        -> plateaued at {best_iter}/{last}; budget adequate.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for a in sys.argv[1:]:
        analyse(a)

"""Did GEPA have enough budget? Parses a sweep log and flags runs that were still improving at the end.

The tell is WHERE in the run the best candidate was found. Found early -> the budget was ample and
the search plateaued. Found in the last quarter -> the search was still climbing when the budget ran
out, and more rollouts would likely have helped.
"""
import re
import sys


def analyse(path: str, label: str = "") -> None:
    txt = open(path, errors="ignore").read().replace("\r", "\n")
    for section in re.split(r"=== penalty ", txt)[1:]:
        lam = section.split()[0]
        iters = [int(m) for m in re.findall(r"Iteration (\d+):", section)]
        if not iters:
            continue
        last = max(iters)
        accepts = [int(m) for m in re.findall(r"Iteration (\d+): New program candidate index", section)]
        bests = [(int(i), float(v)) for i, v in
                 re.findall(r"Iteration (\d+): Best score on valset: ([0-9.]+)", section)]
        base = re.findall(r"Base program full valset score: ([0-9.]+)", section)
        best_iter, best_val = (max(bests, key=lambda t: (t[1], -t[0])) if bests else (None, None))
        ceiling = best_val is not None and best_val >= 0.999
        late = best_iter is not None and last > 0 and best_iter / last > 0.75
        print(f"λ={lam:<5} iterations={last:<4} proposals={len(re.findall(r'Proposed new text', section)):<4}"
              f" accepted={len(accepts):<3} base_val={base[0] if base else '?':<6} best_val={best_val}"
              f" @iter {best_iter}")
        if ceiling:
            print("        -> val CEILING reached; extra budget cannot improve selection. 200 is enough.")
        elif late:
            print(f"        -> best found in the final quarter ({best_iter}/{last}): STILL CLIMBING, "
                  f"raise max_metric_calls.")
        else:
            print(f"        -> plateaued at {best_iter}/{last}; budget adequate.")

if __name__ == "__main__":
    for a in sys.argv[1:]:
        analyse(a)

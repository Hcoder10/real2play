"""A gate consumes paired evaluations, never unrelated export headline scores."""

import math

from .storage import digest


class GateError(ValueError):
    pass


def validate_report(report, plan):
    if report.get("plan_hash") != digest(plan):
        raise GateError("Evaluation does not identify the saved scenes and noise plan")
    if not report.get("policy"):
        raise GateError("Evaluation is missing its policy identity")
    expected = [scene["id"] for scene in plan["episodes"]]
    scores = {}
    for noise in plan["noise"]:
        rows = report.get("results", {}).get(noise, [])
        if [row.get("scene_id") for row in rows] != expected:
            raise GateError(f"{noise}: policies must use IDENTICAL scenes in identical order")
        if [row.get("noise_seed") for row in rows] != [s["noise_seed"] for s in plan["episodes"]]:
            raise GateError(f"{noise}: policies must use IDENTICAL noise seeds")
        if not rows or any(type(row.get("success")) is not bool for row in rows):
            raise GateError(f"{noise}: missing boolean success outcomes")
        scores[noise] = sum(row["success"] for row in rows) / len(rows)
    return scores


def compare(challenger, incumbent, plan, *, min_gain=0.0):
    if not math.isfinite(min_gain) or min_gain < 0:
        raise GateError("min_gain must be finite and nonnegative")
    if not {"clean", "real"}.issubset(plan.get("noise", [])):
        raise GateError("Gate requires clean and real-noise trials")
    candidate = validate_report(challenger, plan)
    baseline = validate_report(incumbent, plan)
    if challenger["policy"] == incumbent["policy"]:
        raise GateError("Challenger and incumbent must be different policies")
    gains = {noise: candidate[noise] - baseline[noise] for noise in plan["noise"]}
    gain = sum(gains.values()) / len(gains)
    # A better average must not hide a regression in a noise condition.
    accepted = gain > min_gain and all(value >= 0 for value in gains.values())
    selected = challenger["policy"] if accepted else incumbent["policy"]
    return {"accepted": accepted, "decision": "PASS" if accepted else "HOLD",
            "selected": selected, "challenger": challenger["policy"],
            "incumbent": incumbent["policy"], "scores": {"challenger": candidate, "incumbent": baseline},
            "gain": gain, "plan_hash": digest(plan), "n": len(plan["episodes"]),
            "message": f"{'PASS' if accepted else 'HOLD'} {challenger['policy']} vs incumbent "
                       f"{incumbent['policy']}; IDENTICAL scenes {digest(plan)[:12]} "
                       f"(n={len(plan['episodes'])}, noise={','.join(plan['noise'])}); selected {selected}"}

"""Illustrative demo data. Never mixed with live adapters or measured results."""

import time


def snapshot():
    return {
        "mock": True, "updated": time.time(),
        "real": {"available": False, "frame": None, "label": "NO REAL FEED", "status": {}, "synthetic": False,
                 "message": "Mock preview • no robot connected", "attempts": [], "argon_says": None},
        "twin": {"available": True, "run": "illustrative_twin", "status": "done", "stage": "done", "synthetic": True,
                 "session_kind": "synthetic", "gap": 0.085, "motion_px": 8.4, "best_iter": 5, "shown_iter": 5,
                 "stop_reason": "plateau", "message": "Best accepted twin · illustrative data", "calls": {},
                 "history": [{"iter": i, "gap": gap, "best_gap": best, "motion_px": px, "kept": keep}
                             for i, gap, best, px, keep in [(0,.48,.48,79,True),(1,.32,.32,51,True),
                                                          (2,.20,.20,30,True),(3,.24,.20,36,False),
                                                          (4,.12,.12,16,True),(5,.085,.085,8.4,True)]],
                 "images": [{"label": label, "src": "/static/mock-bench.svg", "crop": "full"}
                            for label in ["SIM", "REAL", "MOTION"]]},
        "game": {"available": True, "url": None,
                 "groups": {"human": {"rounds": 873, "successes": 694, "hours": 12.4},
                            "bot": {"rounds": 2400, "successes": 2216, "hours": 18.6},
                            "expert": {"rounds": 0, "successes": 0, "hours": 0},
                            "other": {"rounds": 0, "successes": 0, "hours": 0}},
                 "levels": [{"name": name, "human": n, "bot": 400, "other": 0, "successes": int(n*.8)} for name,n in
                            [("cube_in_bowl", 248), ("pick_place_stick", 171), ("stack_two_cubes", 139),
                             ("push_to_target", 126), ("two_cubes_in_bowl", 89), ("place_on_cell", 100)]],
                 "leaderboard": [{"player": p, "successes": n, "rounds": n + 4} for p,n in
                                 [("Mika",42),("Sofia",38),("Alex",35)]], "invalid_traces": 0},
        "policy": {"available": True, "name": "illustrative_policy", "clean": .93, "real": .81, "n": 100,
                   "message": "Illustrative data • not a measured policy", "evidence": "MOCK EVALUATION",
                   "scaling": [{"episodes": n, "clean": p, "real": max(0,p-.12)} for n,p in
                               [(50,.53),(100,.68),(200,.82),(400,.89),(873,.93)]]},
        "ttt": {"available": True, "board": ["X","O","","","X","","O","","O"], "move": "r2c2",
                "why": "Block the diagonal, then keep two ways to win.", "model": "Illustrative Argon response",
                "source": "mock", "synthetic": True, "label": "MOCK BOARD"},
        "runs": [{"schema":"r2p_run/1", "run":"illustrative-dry", "task":"cube_in_bowl", "mode":"dry-run",
                  "synthetic":True, "status":"BLOCKED", "label":"SIM DRY RUN", "rounds":{"r1":{"stages":{
                      "session":{"status":"DONE","message":"Illustrative synthetic attempt"},
                      "train":{"status":"BLOCKED","message":"Example: waiting for a landlord lease"}}}}}]
    }

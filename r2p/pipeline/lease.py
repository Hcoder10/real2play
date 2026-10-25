"""Validate a landlord-issued lease before a training process is started."""

from datetime import datetime, timezone
import json
import urllib.request

from .engine import Blocked
from .storage import read_json

LEDGER = "http://100.112.46.57:7788/api/state"


def lease_step(task, round_name):
    purpose = f"Real2Play {task} {round_name}: policy training (one worker, 4 GiB cap)"
    return ("Training needs an active landlord lease in runs/lease.json. Exact steps: "
            "landlord_overview(); confirm trust=live, rentable=true, unleased_used_gb=0; "
            "rent_apartment(pct=40, vram_gb=4, minutes=30, purpose=" + json.dumps(purpose) +
            "); save the returned lease_id, env, expiry and VRAM cap to runs/lease.json; "
            "rerun this command. Extend before expiry; move_out(lease_id, outcome) when finished.")


def _records(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _records(child)


def _future(value):
    if isinstance(value, (int, float)):
        return value > datetime.now(timezone.utc).timestamp()
    try:
        when = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return when.tzinfo is not None and when > datetime.now(timezone.utc)
    except ValueError:
        return False


def validate_lease(path, *, task, round_name, fetch=None):
    hint = lease_step(task, round_name)
    try:
        lease = read_json(path)
    except (OSError, ValueError) as exc:
        raise Blocked(hint) from exc
    lease = lease.get("lease", lease)
    lease_id = lease.get("lease_id", lease.get("id"))
    env = lease.get("env", {})
    if not lease_id or str(env.get("LANDLORD_LEASE_ID")) != str(lease_id):
        raise Blocked("Lease file must contain landlord's matching LANDLORD_LEASE_ID env. " + hint)
    if not env.get("CUDA_VISIBLE_DEVICES"):
        raise Blocked("Lease is missing landlord's CUDA_VISIBLE_DEVICES. " + hint)
    try:
        if fetch:
            ledger = fetch()
        else:
            with urllib.request.urlopen(LEDGER, timeout=4) as response:
                ledger = json.load(response)
    except (OSError, ValueError) as exc:
        raise Blocked("Cannot verify the live landlord ledger; training was not launched. " + hint) from exc
    active = None
    for row in _records(ledger):
        if str(row.get("lease_id", row.get("id", ""))) == str(lease_id):
            expiry = row.get("expires_at", row.get("expires", row.get("expiry")))
            if row.get("status", "active") in {"active", "rented"} and _future(expiry):
                active = row
                break
    if active is None:
        raise Blocked(f"Landlord lease {lease_id} is absent, inactive or expired. " + hint)
    vram = active.get("vram_gb", lease.get("vram_gb"))
    if not isinstance(vram, (int, float)) or vram <= 0:
        raise Blocked("Lease has no positive VRAM cap; request a complete rent_apartment receipt. " + hint)
    # Do not copy arbitrary receipt environment into the process (e.g. PYTHONPATH).
    allowed = {key: str(value) for key, value in env.items()
               if key.startswith("LANDLORD_") or key in {"CUDA_VISIBLE_DEVICES", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"}}
    allowed["R2P_LEASE_VRAM_GB"] = str(vram)
    return allowed

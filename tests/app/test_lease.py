from datetime import datetime, timedelta, timezone

import pytest

from r2p.pipeline.engine import Blocked
from r2p.pipeline.lease import validate_lease
from r2p.pipeline.storage import atomic_json


def receipt(tmp_path):
    path = tmp_path / "lease.json"
    atomic_json(path, {"lease_id":"lease-fixture", "vram_gb":4,
                      "env":{"LANDLORD_LEASE_ID":"lease-fixture", "CUDA_VISIBLE_DEVICES":"0",
                             "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE":"40", "PYTHONPATH":"untrusted"}})
    return path


def ledger(expired=False):
    return {"leases":[{"id":"lease-fixture", "status":"active", "vram_gb":4,
                       "expires_at":(datetime.now(timezone.utc)+timedelta(minutes=-1 if expired else 30)).isoformat()}]}


def test_verified_receipt_copies_only_leased_environment(tmp_path):
    env = validate_lease(receipt(tmp_path), task="task", round_name="r1", fetch=ledger)
    assert env["R2P_LEASE_VRAM_GB"] == "4" and env["CUDA_VISIBLE_DEVICES"] == "0"
    assert "PYTHONPATH" not in env


@pytest.mark.parametrize("kind", ["expired", "missing", "unreachable"])
def test_invalid_live_lease_blocks_before_launch(tmp_path, kind):
    def fetch():
        if kind == "unreachable": raise OSError("offline")
        return ledger(True) if kind == "expired" else {}
    with pytest.raises(Blocked, match="rent_apartment"):
        validate_lease(receipt(tmp_path), task="task", round_name="r1", fetch=fetch)

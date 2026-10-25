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


def test_actual_landlord_expiry_field_and_nested_receipt(tmp_path):
    path=receipt(tmp_path)
    from r2p.pipeline.storage import read_json
    saved=read_json(path)
    atomic_json(path,{"structuredContent":{"result":{"lease":{"id":saved["lease_id"],"vram_gb":4},"env":saved["env"]}}})
    state=ledger()
    row=state["leases"][0]
    row["expires_time"]=row.pop("expires_at")
    row["env"]={"CUDA_VISIBLE_DEVICES":"2","LANDLORD_LEASE_ID":"lease-fixture"}
    env=validate_lease(path,task="task",round_name="r1",fetch=lambda:state)
    assert env["CUDA_VISIBLE_DEVICES"]=="2"  # live ledger, not stale receipt, determines the GPU


def test_landlord_local_naive_time_is_compared_to_ledger_clock(tmp_path):
    state={"time":"2026-10-25T10:00:00", "active_leases":[{
        "id":"lease-fixture", "status":"active", "expires_time":"2026-10-25T10:30:00", "vram_gb":4}]}
    env=validate_lease(receipt(tmp_path),task="task",round_name="r1",fetch=lambda:state)
    assert env["LANDLORD_LEASE_ID"]=="lease-fixture"
    state["active_leases"][0]["expires_time"]="2026-10-25T09:59:00"
    with pytest.raises(Blocked):
        validate_lease(receipt(tmp_path),task="task",round_name="r1",fetch=lambda:state)


def test_training_worker_applies_vram_cap_without_starting_real_torch(monkeypatch):
    import sys
    from types import SimpleNamespace
    from r2p.pipeline import train_worker
    calls=[]
    cuda=SimpleNamespace(device_count=lambda:1,get_device_properties=lambda i:SimpleNamespace(total_memory=16*1024**3),
                         set_per_process_memory_fraction=lambda fraction,index:calls.append((fraction,index)))
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=cuda))
    monkeypatch.setenv("LANDLORD_LEASE_ID","fixture")
    monkeypatch.setenv("R2P_LEASE_VRAM_GB","4")
    monkeypatch.delenv("LANDLORD_MEMORY_FRACTION",raising=False)
    monkeypatch.setattr(sys,"argv",["worker","--device","cuda"])
    monkeypatch.setattr(train_worker.runpy,"run_module",lambda name,run_name:calls.append((name,list(sys.argv))))
    train_worker.main()
    assert calls[0]==(.25,0)
    assert calls[1][0]=="r2p.policy.train" and calls[1][1][-2:]==["--memory-fraction","0.25"]

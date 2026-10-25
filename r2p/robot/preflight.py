"""Read-only bench diagnostics. Each check emits PASS, WARN or FAIL."""
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from r2p.frames import JOINTS
from .config import ROOT, config_path, load_joint_map


class Report:
    def __init__(self):
        self.lines = []

    def add(self, status, name, detail=""):
        self.lines.append((status, name, str(detail)))

    @property
    def failed(self):
        return any(status == "FAIL" for status, _, _ in self.lines)

    def emit(self):
        for status, name, detail in self.lines:
            print(f"{status} {name}: {detail}")


def check_bus(bus, calibration, report):
    motors = bus.bus.motors
    report.add("PASS" if set(motors) == set(JOINTS) and len(motors) == 6 else "FAIL", "motors", f"{len(motors)} configured motors; handshake verifies each ID")
    for register, field in (("Homing_Offset", "homing_offset"), ("Min_Position_Limit", "range_min"), ("Max_Position_Limit", "range_max")):
        values = bus.read_register(register)
        mismatches = []
        for joint in JOINTS:
            expected = calibration.get(joint, {})
            if is_dataclass(expected):
                expected = asdict(expected)
            if field not in expected or values.get(joint) != expected[field]:
                mismatches.append(joint)
        report.add("FAIL" if mismatches else "PASS", register, "mismatch: " + ", ".join(mismatches) if mismatches else "EEPROM matches calibration file")
    voltage = {k: float(v)/10 for k, v in bus.read_register("Present_Voltage").items()}
    report.add("PASS" if len(voltage) == 6 and all(11 <= v <= 13.5 for v in voltage.values()) else "FAIL", "supply", voltage)
    temperature = bus.read_register("Present_Temperature")
    report.add("FAIL" if len(temperature) != 6 or any(v >= 65 for v in temperature.values()) else "WARN" if any(v >= 50 for v in temperature.values()) else "PASS", "temperature_C", temperature)
    torque = bus.read_register("Torque_Enable")
    report.add("FAIL" if len(torque) != 6 else "PASS" if len(set(torque.values())) == 1 else "WARN", "torque", torque)
    errors = bus.read_register("Status")
    report.add("FAIL" if len(errors) != 6 or any(errors.values()) else "PASS", "motor_error_flags", errors)


def run_preflight(cfg, *, policy=None, tests=True, allow_unverified=False, bus=None, camera=None, mock=False):
    report = Report()
    report.add("PASS", "config", f"30 Hz={cfg.get('control_hz') == 30}; platform overlay loaded")
    if cfg.get("control_hz") != 30:
        report.add("FAIL", "control_hz", "must equal 30")
    if mock:
        report.add("WARN", "hardware", "mock mode: port, EEPROM, supply, temperature and camera were not checked")
    else:
        own_bus = bus is None
        try:
            if bus is None:
                from .io import ReadOnlyBus
                bus = ReadOnlyBus(cfg)
            report.add("PASS", "port", cfg.get("port"))
            path = Path(cfg["calibration_dir"]).expanduser() / f"{cfg.get('robot_id', 'follower_1')}.json"
            calibration = json.loads(path.read_text(encoding="utf-8"))
            check_bus(bus, calibration, report)
        except Exception as exc:
            report.add("FAIL", "bus_or_calibration", str(exc))
        finally:
            if own_bus and bus is not None:
                bus.close()
        own_camera = camera is None
        try:
            if camera is None:
                from r2p.perception import open_camera
                camera = open_camera(cfg["camera"])
            frame = camera.read()
            size = frame.shape[1::-1]
            expected = (cfg["camera"]["width"], cfg["camera"]["height"])
            report.add("PASS" if size == expected else "FAIL", "camera_size", f"{size}, expected {expected}")
            # Count distinct capture timestamps (or image bytes for simple sources).
            stamps = set()
            started = time.monotonic()
            for _ in range(30):
                current = camera.read()
                stamp = getattr(camera, "_captured", None)
                stamps.add(stamp if stamp is not None else hash(current.tobytes()))
                time.sleep(1/30)
            fps = len(stamps)/(time.monotonic()-started)
            report.add("PASS" if fps >= 20 else "WARN" if fps >= 5 else "FAIL", "camera_fps", f"{fps:.1f} fresh frames/s")
            age = getattr(camera, "age_s", 0)
            report.add("PASS" if age < 1 else "FAIL", "camera_age_s", age)
            capture = getattr(camera, "capture", None)
            if capture is not None:
                import cv2
                report.add("PASS", "camera_exposure", capture.get(cv2.CAP_PROP_EXPOSURE))
            else:
                report.add("WARN", "camera_exposure", "HTTP source does not expose sensor exposure; inspect camserver")
        except Exception as exc:
            report.add("FAIL", "camera", str(exc))
        finally:
            if own_camera and camera is not None:
                camera.close()
    try:
        from r2p.perception import CameraModel
        path = config_path(cfg["camera"]["calibration"])
        model = CameraModel.load(path)
        good = model.image_size == (cfg["camera"]["width"], cfg["camera"]["height"])
        report.add("PASS" if good else "FAIL", "camera_calibration", str(path))
    except Exception as exc:
        report.add("WARN" if mock else "FAIL", "camera_calibration", str(exc))
    report.add("PASS" if cfg.get("board") else "FAIL", "board", "configured" if cfg.get("board") else "missing")
    try:
        jm = load_joint_map(cfg)
        report.add("PASS" if jm.verified else "WARN" if allow_unverified or mock else "FAIL", "joint_map", f"{jm.name}, verified={jm.verified}")
    except Exception as exc:
        report.add("FAIL", "joint_map", str(exc))
    if policy is not None:
        try:
            from .policy import check_chunk, policy_config
            from r2p.tasks import load_task, sample_scene, encode_slots, commands
            p_cfg = policy_config(policy)
            task = load_task(p_cfg["task"])
            scene = sample_scene(task, np.random.default_rng(0))
            poses = {o["id"]: o for o in scene["objects"]}
            check_chunk(policy.act({"q": np.asarray(p_cfg["start_q"]), "slots": encode_slots(task, poses), "command": commands(task)[0]}))
            report.add("PASS", "policy", "loads and returns finite (15, 6) actions")
        except Exception as exc:
            report.add("FAIL", "policy", str(exc))
    else:
        report.add("WARN", "policy", "no export specified")
    if tests:
        try:
            result = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/robot"], cwd=ROOT, capture_output=True, text=True, timeout=180)
            report.add("PASS" if result.returncode == 0 else "FAIL", "pytest", result.stdout.strip().splitlines()[-1] if result.stdout.strip() else result.stderr[-500:])
        except (OSError, subprocess.TimeoutExpired) as exc:
            report.add("FAIL", "pytest", str(exc))
    else:
        report.add("WARN", "pytest", "skipped")
    return report

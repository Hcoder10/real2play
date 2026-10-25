import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from r2p.perception.camera import CameraModel, HttpCamera, LiveCamera, StillCamera, open_camera


def test_distorted_deprojection_accuracy_and_json(tmp_path):
    cam = CameraModel.look_at(image_size=(1280, 720))
    cam.dist = np.array([-.12, .035, .001, -.002, 0])
    rng = np.random.default_rng(12)
    for h in (0, .03, .11):
        points = np.column_stack((rng.uniform(.08, .35, 100), rng.uniform(-.18, .18, 100), np.full(100, h)))
        recovered = cam.deproject(cam.project(points), h=h)
        assert np.max(np.linalg.norm(recovered-points, axis=1)) < .002
    path = tmp_path / "camera.json"
    cam.save(path)
    loaded = CameraModel.load(path)
    assert loaded.to_dict() == cam.to_dict()
    assert np.allclose(loaded.position, cam.position)
    small = cam.resized((640, 360))
    assert np.allclose(small.project(points), cam.project(points)/2)


def test_parallel_and_behind_plane():
    cam = CameraModel(np.eye(3))
    with pytest.raises(ValueError, match="behind"):
        cam.deproject([0, 0], -1)
    side = CameraModel.look_at((0, 0, .2), (1, 0, .2))
    with pytest.raises(ValueError, match="parallel"):
        side.deproject([side.width/2, side.height/2])


def test_still_and_http_camera(tmp_path):
    frame = np.full((40, 80, 3), (12, 80, 220), np.uint8)
    path = tmp_path / "still.png"
    cv2.imwrite(str(path), frame)
    with open_camera({"source": "still", "path": path, "width": 160, "height": 90}) as camera:
        assert camera.read().shape == (90, 160, 3)
        assert camera.age_s == 0
    encoded = cv2.imencode(".jpg", frame)[1].tobytes()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("X-Frame-Age", "1.25")
            self.end_headers()
            self.wfile.write(encoded)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with HttpCamera({"url": f"http://127.0.0.1:{server.server_port}/frame/0.jpg", "width": 160, "height": 90}) as camera:
            assert camera.read().shape == (90, 160, 3)
            assert camera.age_s >= 1.25
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_live_background_capture_uses_configured_backend_without_hardware(monkeypatch):
    captured = []
    class FakeCapture:
        released = False
        def __init__(self, index, backend):
            captured.append((index, backend))
        def isOpened(self):
            return True
        def set(self, *args):
            pass
        def read(self):
            return True, np.full((30, 40, 3), 90, np.uint8)
        def release(self):
            self.released = True
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    with LiveCamera({"index": 3, "backend": "ANY", "width": 80, "height": 60}) as camera:
        assert camera.read().shape == (60, 80, 3)
        assert camera.age_s < 1
        assert camera._thread.is_alive()
    assert captured == [(3, cv2.CAP_ANY)]
    assert camera.capture.released
    assert not camera._thread.is_alive()


def test_camserver_list_and_frame_routes():
    from urllib.request import urlopen
    from scripts.perception.camserver import make_handler
    camera = StillCamera(np.full((720, 1280, 3), 80, np.uint8))
    camera.error = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler({2: camera}))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base+"/list") as response:
            assert json.load(response)[0]["index"] == 2
        with urlopen(base+"/frame/2.jpg") as response:
            image = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
            assert image.shape == (720, 1280, 3)
            assert float(response.headers["X-Frame-Age"]) == 0
    finally:
        server.shutdown()
        server.server_close()
        worker.join()

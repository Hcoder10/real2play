import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from r2p.perception.camera import CameraModel, HttpCamera, StillCamera, open_camera


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

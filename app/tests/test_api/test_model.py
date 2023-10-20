"""Model Detection Unit Test"""
# pylint: disable=unused-import

from __future__ import annotations

import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_detect_error():
    """Test detection error"""
    response = client.post("aimodel/deepfake_detection")
    assert response.status_code == 422

def test_detect_success():
    """Test detection success"""
    with client:
        image_path = "app/tests/example_data/HarryOnana.jpg"

        with open(image_path, "rb") as image_file:
            _file = {"file": ("filename", image_file, "image/jpeg")}
            response = client.post("aimodel/deepfake_detection", files=_file)

            assert response.status_code == 200
            # data = response.json()
            # assert "Deepfake" in data

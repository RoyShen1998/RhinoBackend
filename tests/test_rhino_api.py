import base64
import json

import index  # the Lambda module; made importable by pytest.ini
import pytest
from botocore.exceptions import ClientError

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
MODEL_ARN = "arn:aws:rekognition:ap-southeast-2:123456789012:project/rhinos/version/v1/1700000000000"


class FakeRekognition:
    def __init__(self):
        self.calls = []
        self.error_code = None
        self.model_status = "RUNNING"

    def _call(self, name, kwargs, response):
        self.calls.append((name, kwargs))
        if self.error_code:
            raise ClientError({"Error": {"Code": self.error_code, "Message": "boom"}}, name)
        return response

    def detect_labels(self, **kwargs):
        return self._call("detect_labels", kwargs, {
            "Labels": [
                {"Name": "Animal", "Confidence": 80.0},
                {"Name": "Rhino", "Confidence": 99.0},
            ],
            "ResponseMetadata": {"RequestId": "abc"},
        })

    def detect_custom_labels(self, **kwargs):
        return self._call("detect_custom_labels", kwargs, {
            "CustomLabels": [
                {"Name": "black_female", "Confidence": 60.0},
                {"Name": "white_male", "Confidence": 90.0},
                {"Name": "white_male", "Confidence": 70.0},
            ],
        })

    def describe_projects(self, **kwargs):
        return self._call("describe_projects", kwargs, {
            "ProjectDescriptions": [{"ProjectArn": "arn:project"}],
        })

    def describe_project_versions(self, **kwargs):
        return self._call("describe_project_versions", kwargs, {
            "ProjectVersionDescriptions": [{"Status": self.model_status}],
        })


@pytest.fixture
def rekognition(monkeypatch):
    fake = FakeRekognition()
    monkeypatch.setattr(index, "get_client", lambda: fake)
    for name in ("REKOGNITION_MODEL_ARN", "RHINO_TYPES", "RHINO_GENDERS", "MIN_CONFIDENCE"):
        monkeypatch.delenv(name, raising=False)
    return fake


def call(method, path, body=None):
    event = {
        "rawPath": path,
        "requestContext": {"http": {"method": method}},
        "body": json.dumps(body) if body is not None else None,
    }
    response = index.lambda_handler(event, None)
    return response["statusCode"], json.loads(response["body"])


def image_body(data=JPEG, **extra):
    return {"image": base64.b64encode(data).decode(), **extra}


def test_filters(rekognition):
    status, body = call("GET", "/rhino-filters")
    assert status == 200
    assert body == {
        "types": ["white", "black", "indian", "javan", "sumatran"],
        "genders": ["male", "female"],
    }


def test_trailing_slash_and_unknown_routes(rekognition):
    assert call("GET", "/rhino-filters/")[0] == 200
    assert call("GET", "/nope")[0] == 404
    assert call("GET", "/analyze-image")[0] == 405


def test_health_general_mode(rekognition):
    status, body = call("GET", "/health")
    assert status == 200
    assert body["mode"] == "general"
    assert body["rekognition"]["modelStatus"] == "GENERAL_LABELS"
    assert rekognition.calls == []


def test_health_custom_mode(rekognition, monkeypatch):
    monkeypatch.setenv("REKOGNITION_MODEL_ARN", MODEL_ARN)
    _, body = call("GET", "/health")
    assert body["status"] == "ok"
    assert body["mode"] == "custom"
    assert body["rekognition"]["modelStatus"] == "RUNNING"
    assert rekognition.calls[0] == ("describe_projects", {"ProjectNames": ["rhinos"]})
    assert rekognition.calls[1] == (
        "describe_project_versions", {"ProjectArn": "arn:project", "VersionNames": ["v1"]},
    )

    rekognition.model_status = "STOPPED"
    _, body = call("GET", "/health")
    assert body["status"] == "degraded"
    assert body["rekognition"]["modelStatus"] == "STOPPED"


def test_health_reports_unreachable_rekognition(rekognition, monkeypatch):
    monkeypatch.setenv("REKOGNITION_MODEL_ARN", MODEL_ARN)
    rekognition.error_code = "AccessDeniedException"
    status, body = call("GET", "/health")
    assert status == 200
    assert body["rekognition"]["connected"] is False


def test_analyze_general_mode(rekognition):
    status, body = call("POST", "/analyze-image", image_body())
    assert status == 200
    assert body["mode"] == "general"
    assert body["rhinoDetected"] is True
    assert [label["Name"] for label in body["sortedLabels"]] == ["Rhino", "Animal"]
    assert body["imageInfo"] == {"size": len(JPEG), "mimeType": "image/jpeg"}
    assert "ResponseMetadata" not in body["rawResponse"]
    name, kwargs = rekognition.calls[0]
    assert name == "detect_labels"
    assert kwargs["Image"] == {"Bytes": JPEG}


def test_analyze_custom_mode_flags_mismatches(rekognition, monkeypatch):
    monkeypatch.setenv("REKOGNITION_MODEL_ARN", MODEL_ARN)
    status, body = call(
        "POST", "/analyze-image",
        image_body(PNG, rhinoType="white", rhinoGender="male"),
    )
    assert status == 200
    assert body["mode"] == "custom"
    assert body["imageInfo"]["mimeType"] == "image/png"
    assert rekognition.calls[0][1]["ProjectVersionArn"] == MODEL_ARN

    # Duplicate label names collapse to the highest confidence, sorted descending.
    top, second = body["sortedLabels"]
    assert (top["Name"], top["Confidence"]) == ("white_male", 90.0)
    assert (top["type"], top["gender"]) == ("white", "male")
    assert top["matches"] == {"type": True, "gender": True}
    assert (second["type"], second["gender"]) == ("black", "female")
    assert second["matches"] == {"type": False, "gender": False}


def test_analyze_accepts_data_url(rekognition):
    body = {"image": "data:image/jpeg;base64," + base64.b64encode(JPEG).decode()}
    assert call("POST", "/analyze-image", body)[0] == 200


@pytest.mark.parametrize("body, expected", [
    (None, 400),
    ({}, 400),
    ({"image": "not base64!"}, 400),
    ({"image": base64.b64encode(b"GIF89a" + b"\x00" * 16).decode()}, 400),
    (image_body(rhinoType="pink"), 400),
    (image_body(rhinoGender=5), 400),
    (image_body(b"\xff\xd8\xff" + b"\x00" * index.MAX_IMAGE_BYTES), 413),
])
def test_analyze_rejects_bad_requests(rekognition, body, expected):
    status, response = call("POST", "/analyze-image", body)
    assert status == expected
    assert response["message"]
    assert rekognition.calls == []


@pytest.mark.parametrize("code, expected_status, expected_body_status", [
    ("ResourceNotReadyException", 503, "MODEL_NOT_READY"),
    ("InvalidImageFormatException", 400, "INVALID_IMAGE"),
    ("ThrottlingException", 429, "THROTTLED"),
    ("SomethingElse", 502, "REKOGNITION_ERROR"),
])
def test_analyze_translates_rekognition_errors(rekognition, code, expected_status, expected_body_status):
    rekognition.error_code = code
    status, body = call("POST", "/analyze-image", image_body())
    assert status == expected_status
    assert body["status"] == expected_body_status


def test_unhandled_errors_do_not_leak(rekognition, monkeypatch):
    def boom(_event):
        raise RuntimeError("secret detail")
    monkeypatch.setitem(index.ROUTES, ("GET", "/health"), boom)
    status, body = call("GET", "/health")
    assert status == 500
    assert "secret" not in json.dumps(body)


@pytest.mark.parametrize("name, expected", [
    ("white_male", ("white", "male")),
    ("Black Rhino Female", ("black", "female")),
    ("SumatranRhinoFemale", ("sumatran", "female")),
    ("Rhino", (None, None)),
])
def test_label_name_parsing(rekognition, name, expected):
    label = index.build_labels([{"Name": name, "Confidence": 50}])[0]
    assert (label["type"], label["gender"]) == expected

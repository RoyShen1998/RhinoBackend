"""Rhino identification API.

Runs behind a Lambda Function URL (payload format 2.0) and exposes:

    GET  /health         Rekognition / model status
    GET  /rhino-filters  rhino types and genders the UI can filter on
    POST /analyze-image  {"image": <base64>, "rhinoType"?, "rhinoGender"?}

Two modes, chosen by the REKOGNITION_MODEL_ARN environment variable:

* custom  - the variable holds a Rekognition Custom Labels project version ARN
            and images are classified with DetectCustomLabels.
* general - the variable is empty and images go through the stock DetectLabels
            API, which can say "this is a rhino" but not which kind.

CORS is handled by the Function URL, so no CORS headers are set here.
"""
import base64
import binascii
import json
import logging
import os
import re

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Rekognition rejects inline images larger than 5 MB.
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_GENERAL_LABELS = 25

MODEL_ARN_PATTERN = re.compile(
    r"^arn:[^:]+:rekognition:[^:]+:\d+:project/(?P<project>[^/]+)/version/(?P<version>[^/]+)/\d+$"
)

# Rekognition error code -> (HTTP status, status string, message)
REKOGNITION_ERRORS = {
    "ResourceNotReadyException": (
        503, "MODEL_NOT_READY",
        "The rhino recognition model is not running yet. Start it and try again in a few minutes.",
    ),
    "ResourceNotFoundException": (
        503, "MODEL_NOT_FOUND",
        "The configured rhino recognition model does not exist.",
    ),
    "ImageTooLargeException": (413, "IMAGE_TOO_LARGE", "The image is too large to analyse."),
    "InvalidImageFormatException": (400, "INVALID_IMAGE", "The image must be a JPEG or PNG."),
    "InvalidParameterException": (400, "INVALID_REQUEST", "Rekognition rejected the image."),
    "ThrottlingException": (429, "THROTTLED", "Too many requests. Please try again shortly."),
    "ProvisionedThroughputExceededException": (
        429, "THROTTLED", "Too many requests. Please try again shortly.",
    ),
    "LimitExceededException": (429, "THROTTLED", "Too many requests. Please try again shortly."),
}

_client = None


class ApiError(Exception):
    def __init__(self, http_status, message, status="ERROR"):
        super().__init__(message)
        self.http_status = http_status
        self.message = message
        self.status = status


def get_client():
    global _client
    if _client is None:
        _client = boto3.client("rekognition")
    return _client


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def _csv_env(name, default):
    values = [v.strip().lower() for v in os.environ.get(name, default).split(",")]
    return [v for v in values if v]


def rhino_types():
    return _csv_env("RHINO_TYPES", "white,black,indian,javan,sumatran")


def rhino_genders():
    return _csv_env("RHINO_GENDERS", "male,female")


def model_arn():
    return os.environ.get("REKOGNITION_MODEL_ARN", "").strip()


def min_confidence():
    try:
        return min(100.0, max(0.0, float(os.environ.get("MIN_CONFIDENCE", "50"))))
    except ValueError:
        return 50.0


# --------------------------------------------------------------------------
# Label handling
# --------------------------------------------------------------------------

def _tokens(name):
    """'WhiteRhino_female' -> ['white', 'rhino', 'female']"""
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    return [t for t in re.split(r"[^a-z]+", spaced.lower()) if t]


def _first_known(tokens, known):
    return next((t for t in tokens if t in known), None)


def build_labels(detected, rhino_type=None, rhino_gender=None):
    """Turn Rekognition labels into the sorted, annotated list the UI shows.

    A label's rhino type and gender are read from its name, so a custom model
    should name its labels along the lines of "white_male" or "Black Rhino
    Female". A label is flagged as a mismatch when a filter was requested and
    the label names a different type/gender.
    """
    best = {}
    for label in detected:
        name = label.get("Name")
        if not name:
            continue
        confidence = float(label.get("Confidence", 0.0))
        if name not in best or confidence > best[name]:
            best[name] = confidence

    types, genders = rhino_types(), rhino_genders()
    labels = []
    for name, confidence in best.items():
        tokens = _tokens(name)
        label_type = _first_known(tokens, types)
        label_gender = _first_known(tokens, genders)
        labels.append({
            "Name": name,
            "Confidence": confidence,
            "type": label_type,
            "gender": label_gender,
            "matches": {
                "type": not rhino_type or label_type is None or label_type == rhino_type,
                "gender": not rhino_gender or label_gender is None or label_gender == rhino_gender,
            },
        })
    # Highest confidence first; on a tie, rhino labels come before generic ones.
    labels.sort(key=lambda item: (-item["Confidence"], not _is_rhino(item), item["Name"]))
    return labels


def _is_rhino(label):
    return label["type"] is not None or any(t.startswith("rhino") for t in _tokens(label["Name"]))


# --------------------------------------------------------------------------
# Request helpers
# --------------------------------------------------------------------------

def _json_body(event):
    body = event.get("body")
    if not body:
        raise ApiError(400, "Request body is required.", "INVALID_REQUEST")
    try:
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body)
        payload = json.loads(body)
    except (ValueError, binascii.Error):
        raise ApiError(400, "Request body must be valid JSON.", "INVALID_REQUEST")
    if not isinstance(payload, dict):
        raise ApiError(400, "Request body must be a JSON object.", "INVALID_REQUEST")
    return payload


def _mime_type(data):
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    return None


def _decode_image(value):
    if not isinstance(value, str) or not value.strip():
        raise ApiError(400, "'image' must be a base64 encoded JPEG or PNG.", "INVALID_REQUEST")
    encoded = re.sub(r"^data:[^;,]+;base64,", "", value.strip())
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise ApiError(400, "'image' is not valid base64.", "INVALID_REQUEST")
    if len(data) > MAX_IMAGE_BYTES:
        raise ApiError(413, "The image is too large to analyse (5 MB maximum).", "IMAGE_TOO_LARGE")
    mime_type = _mime_type(data)
    if mime_type is None:
        raise ApiError(400, "The image must be a JPEG or PNG.", "INVALID_IMAGE")
    return data, mime_type


def _filter_value(payload, key, allowed):
    value = payload.get(key)
    if value is None or value == "":
        return None
    if not isinstance(value, str) or value.lower() not in allowed:
        raise ApiError(400, f"'{key}' must be one of: {', '.join(allowed)}.", "INVALID_REQUEST")
    return value.lower()


def _rekognition_error(error):
    code = error.response.get("Error", {}).get("Code", "")
    logger.warning("Rekognition call failed: %s", error)
    if code in REKOGNITION_ERRORS:
        http_status, status, message = REKOGNITION_ERRORS[code]
        return ApiError(http_status, message, status)
    return ApiError(502, "The recognition service returned an error.", "REKOGNITION_ERROR")


def _without_metadata(response):
    return {k: v for k, v in response.items() if k != "ResponseMetadata"}


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

def get_filters(_event):
    return 200, {"types": rhino_types(), "genders": rhino_genders()}


def _model_status(arn):
    """Return (status, message) for a Custom Labels project version."""
    match = MODEL_ARN_PATTERN.match(arn)
    if not match:
        return "INVALID_MODEL_ARN", "REKOGNITION_MODEL_ARN is not a project version ARN."
    client = get_client()
    projects = client.describe_projects(ProjectNames=[match["project"]]).get("ProjectDescriptions", [])
    if not projects:
        return "MODEL_NOT_FOUND", "The Rekognition project does not exist."
    versions = client.describe_project_versions(
        ProjectArn=projects[0]["ProjectArn"], VersionNames=[match["version"]]
    ).get("ProjectVersionDescriptions", [])
    if not versions:
        return "MODEL_NOT_FOUND", "The Rekognition project version does not exist."
    status = versions[0].get("Status", "UNKNOWN")
    return status, versions[0].get("StatusMessage") or f"Model status: {status}"


def get_health(_event):
    arn = model_arn()
    if not arn:
        return 200, {
            "status": "ok",
            "mode": "general",
            "rekognition": {
                "connected": True,
                "message": "No custom model configured; using general Rekognition label detection.",
                "modelStatus": "GENERAL_LABELS",
            },
        }
    try:
        model_status, message = _model_status(arn)
        connected = True
    except (ClientError, BotoCoreError) as error:
        logger.warning("Model status check failed: %s", error)
        model_status, message, connected = "UNKNOWN", "Could not read the model status.", False
    return 200, {
        "status": "ok" if model_status == "RUNNING" else "degraded",
        "mode": "custom",
        "rekognition": {"connected": connected, "message": message, "modelStatus": model_status},
    }


def analyze_image(event):
    payload = _json_body(event)
    image, mime_type = _decode_image(payload.get("image"))
    rhino_type = _filter_value(payload, "rhinoType", rhino_types())
    rhino_gender = _filter_value(payload, "rhinoGender", rhino_genders())

    arn = model_arn()
    client = get_client()
    try:
        if arn:
            raw = client.detect_custom_labels(
                ProjectVersionArn=arn, Image={"Bytes": image}, MinConfidence=min_confidence()
            )
            detected = raw.get("CustomLabels", [])
        else:
            raw = client.detect_labels(
                Image={"Bytes": image}, MaxLabels=MAX_GENERAL_LABELS, MinConfidence=min_confidence()
            )
            detected = raw.get("Labels", [])
    except ClientError as error:
        raise _rekognition_error(error)
    except BotoCoreError as error:
        logger.warning("Rekognition call failed: %s", error)
        raise ApiError(502, "Could not reach the recognition service.", "REKOGNITION_ERROR")

    labels = build_labels(detected, rhino_type, rhino_gender)
    return 200, {
        "mode": "custom" if arn else "general",
        "rhinoDetected": any(_is_rhino(label) for label in labels),
        "sortedLabels": labels,
        "imageInfo": {"size": len(image), "mimeType": mime_type},
        "rawResponse": _without_metadata(raw),
    }


ROUTES = {
    ("GET", "/health"): get_health,
    ("GET", "/rhino-filters"): get_filters,
    ("POST", "/analyze-image"): analyze_image,
}


def _response(http_status, body):
    return {
        "statusCode": http_status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body, default=str),
    }


def lambda_handler(event, _context):
    http = event.get("requestContext", {}).get("http", {})
    method = str(http.get("method", "GET")).upper()
    path = "/" + str(event.get("rawPath", "/")).strip("/")

    route = ROUTES.get((method, path))
    if route is None:
        if any(route_path == path for _, route_path in ROUTES):
            return _response(405, {"message": f"{method} is not allowed on {path}.", "status": "METHOD_NOT_ALLOWED"})
        return _response(404, {"message": f"No such route: {path}.", "status": "NOT_FOUND"})

    try:
        return _response(*route(event))
    except ApiError as error:
        return _response(error.http_status, {"message": error.message, "status": error.status})
    except Exception:  # noqa: BLE001 - last line of defence, never leak a stack trace
        logger.exception("Unhandled error on %s %s", method, path)
        return _response(500, {"message": "Internal server error.", "status": "ERROR"})

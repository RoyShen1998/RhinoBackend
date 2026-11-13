"""Serve the rhino API Lambda on http://localhost:3000 for frontend development.

    python local_api.py            # real Rekognition, using your AWS credentials
    python local_api.py --mock     # canned results, no AWS calls

Personal settings (AWS profile, region, Custom Labels model ARN) are read from
a git-ignored .env file next to this script; see .env.example.
"""
import argparse
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent / "rhino_api"))

import index  # noqa: E402

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


class MockRekognition:
    """Stands in for the boto3 client so the UI can be developed offline."""

    def detect_labels(self, **_kwargs):
        return {"Labels": [
            {"Name": "Rhino", "Confidence": 98.7},
            {"Name": "Wildlife", "Confidence": 98.7},
            {"Name": "Animal", "Confidence": 98.7},
            {"Name": "Mammal", "Confidence": 97.2},
            {"Name": "Grassland", "Confidence": 71.4},
        ]}

    def detect_custom_labels(self, **_kwargs):
        return {"CustomLabels": [
            {"Name": "white_male", "Confidence": 91.3},
            {"Name": "white_female", "Confidence": 64.8},
            {"Name": "black_male", "Confidence": 52.1},
        ]}

    def describe_projects(self, **_kwargs):
        return {"ProjectDescriptions": [{"ProjectArn": "arn:aws:rekognition:local:000000000000:project/mock/0"}]}

    def describe_project_versions(self, **_kwargs):
        return {"ProjectVersionDescriptions": [{"Status": "RUNNING", "StatusMessage": "Mock model is running."}]}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, headers, body=b""):
        self.send_response(status)
        for name, value in {**headers, **CORS_HEADERS}.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _dispatch(self):
        length = int(self.headers.get("Content-Length") or 0)
        event = {
            "version": "2.0",
            "rawPath": urlsplit(self.path).path,
            "requestContext": {"http": {"method": self.command}},
            "body": self.rfile.read(length).decode("utf-8") if length else None,
            "isBase64Encoded": False,
        }
        result = index.lambda_handler(event, None)
        self._send(result["statusCode"], result.get("headers", {}), result.get("body", "").encode("utf-8"))

    do_GET = _dispatch
    do_POST = _dispatch

    def do_OPTIONS(self):
        self._send(204, {})


def load_env_file(path):
    """Load KEY=VALUE lines from a git-ignored .env file into the environment.

    Personal settings (AWS profile, model ARN) live there instead of in the
    code. Variables that are already set in the environment win.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--mock", action="store_true", help="return canned results instead of calling AWS")
    args = parser.parse_args()

    load_env_file(Path(__file__).resolve().parent / ".env")

    if args.mock:
        mock = MockRekognition()
        index.get_client = lambda: mock

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Rhino API listening on http://localhost:{args.port} ({'mock' if args.mock else 'AWS'} mode)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

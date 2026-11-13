# Rhino AI - backend

The API behind Rhino AI: it takes a photo and identifies rhinos with AWS Rekognition.
It is a single AWS Lambda function (`rhino_api/index.py`), deployed by the infra repository
and called by the frontend repository.

## API

| Route | Purpose |
| --- | --- |
| `GET /health` | Recognition mode and model status. |
| `GET /rhino-filters` | The rhino types and genders the UI can filter on. |
| `POST /analyze-image` | Body `{"image": "<base64 JPEG or PNG>", "rhinoType"?: "...", "rhinoGender"?: "..."}`. Returns the labels found, sorted by confidence. |

Errors are JSON `{"message": "...", "status": "..."}` with a matching HTTP status.

## Recognition modes

| Mode | When | What you get |
| --- | --- | --- |
| general | `REKOGNITION_MODEL_ARN` is empty (default) | Stock Rekognition `DetectLabels`. Says whether a rhino is in the photo. |
| custom | `REKOGNITION_MODEL_ARN` is a Custom Labels project version ARN | Species and gender, with labels flagged when they differ from the requested type/gender. |

In custom mode the type and gender are read from each label's **name**, so name the model's
labels accordingly, for example `white_male` or `Black Rhino Female`.

Environment variables: `REKOGNITION_MODEL_ARN`, `MIN_CONFIDENCE` (default 50),
`RHINO_TYPES` (default `white,black,indian,javan,sumatran`), `RHINO_GENDERS` (default `male,female`).

## Setup

```
python -m venv .venv
.venv\Scripts\activate.bat            # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Personal settings and credentials

Nothing personal is stored in this repository:

- **Settings** for running locally (AWS profile, region, Custom Labels model ARN) go in `.env`, which is git-ignored. Copy the template and edit it:

  ```
  copy .env.example .env
  ```

- **AWS credentials** never go in any file here. boto3 reads them from your AWS CLI configuration (`aws configure`).

## Run locally

Serves the API on http://localhost:3000, which is where the frontend's dev server looks for it:

```
python local_api.py --mock    # canned results, no AWS needed
python local_api.py           # real Rekognition with your AWS credentials
```

## Test

```
pytest
```

## Deployment

Deployed by `terraform apply` in the infra repository, which zips the `rhino_api/` directory as the Lambda
with handler `index.lambda_handler`. The code may only use the standard library and boto3.

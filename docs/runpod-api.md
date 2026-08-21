# Applio RunPod Serverless API

Applio training runs on RunPod Serverless. Inference runs on the persistent
`voice-gen` pod and is outside this Serverless contract.

| Endpoint | Command | Workers | Timeout |
|---|---|---|---|
| `forge-applio-training` | `python -u -m app.train_handler` | `workersMin = 0` | 6 hours |

## 1. Endpoints

| Use | URL | Notes |
|---|---|---|
| Training (async) | `https://api.runpod.ai/v2/<train-endpoint>/run` | always async (5–30 min) |
| Job status | `https://api.runpod.ai/v2/<endpoint>/status/{job_id}` | progress included for training |

The endpoint id is provisioned out-of-band and passed to `voice_service` via
`APPLIO_TRAINING_ENDPOINT_ID`.

## 2. Common conventions

### Auth

```
Authorization: Bearer <RUNPOD_API_KEY>
Content-Type: application/json
```

### Request body

Wrap in `{"input": {...}}`. Training does not need `action` because the train
handler defaults it to `train` and rejects every other value.

### Response body

```json
{
  "status": "COMPLETED" | "FAILED" | "IN_QUEUE" | "IN_PROGRESS",
  "output": { ... },
  "error": "... (on FAILED)"
}
```

Always check **both** `status == "COMPLETED"` and `output.error`.

## 3. Training (`/run`, always async)

### 3.1 Start

```json
{
  "input": {
    "character_id": "xxx",
    "character_name": "Evie",
    "model_name": "evie_f_xxx",
    "audio_s3_prefix": "s3://bucket/voice/training/<char_id>/<audio_dir>/",
    "training_config": {
      "sample_rate": 40000,
      "total_epoch": 100,
      "batch_size": 4,
      "f0_method": "rmvpe",
      "embedder_model": "contentvec",
      "vocoder": "HiFi-GAN",
      "save_every_epoch": 5,
      "save_every_weights": true,
      "overtraining_detector": true,
      "overtraining_threshold": 50,
      "noise_reduction": true,
      "noise_reduction_strength": 0.7
    },
    "callback_url": "http://<voice_service>/training/callback"
  }
}
```

Response (immediate):

```json
{"id": "<runpod-job-id>", "status": "IN_QUEUE"}
```

### 3.2 Poll

`GET /v2/<train-endpoint>/status/<job_id>`

```json
{
  "id": "...",
  "status": "IN_PROGRESS",
  "output": {
    "stage": "train",
    "step": 0,
    "total_steps": 0,
    "extra": {}
  }
}
```

`stage` progresses through `download_data → preprocess → extract → train →
upload → completed | failed`.

Terminal COMPLETED payload:

```json
{
  "status": "COMPLETED",
  "output": {
    "job_id": "internal-uuid",
    "character_id": "xxx",
    "character_name": "Evie",
    "model_name": "evie_f_xxx",
    "status": "completed",
    "rvc_model_path": "s3://.../logs/<model>/<model>_<n>_best_epoch.pth",
    "rvc_index_path": "s3://.../logs/<model>/<model>.index",
    "error_message": null
  }
}
```

### 3.3 Callback

If `callback_url` is provided, the worker POSTs on completion (success or
failure):

```json
{
  "job_id": "internal-uuid",
  "character_id": "xxx",
  "character_name": "Evie",
  "engine": "applio",
  "status": "completed" | "failed",
  "result": {
    "rvc_model_path": "s3://...",
    "rvc_index_path": "s3://..."
  },
  "error_message": null
}
```

## 4. Environment variables

| Key | Default | Purpose |
|---|---|---|
| `APPLIO_BASE_DIR` | `/app/Applio` | Source repo root |
| `APPLIO_VOLUME_PATH` | `/runpod-volume/applio` | Network Volume mount |
| `APPLIO_WORK_ROOT` | `/tmp/work` | Ephemeral working root |
| `APPLIO_MODEL_REGISTRY_S3` | `s3://shiftup-enterprise-ai-service/voice/model_registry/Applio` | Training upload destination |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION` | — | S3 access |

## 5. Cold start

First worker spinup takes 60–120s (image pull + torch load). Training uses
`workersMin = 0` because the cold start is negligible compared with a full RVC
run.

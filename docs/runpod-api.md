# Applio RunPod Serverless API

Applio migrated from the on-prem FastAPI server (`:6969`) to RunPod
Serverless. This document is the call contract clients (Forge
`voice_service`) need to use.

Two endpoints share a single image; `dockerArgs` selects the handler module:

| Endpoint | Command | Workers | Timeout |
|---|---|---|---|
| `forge-applio-inference` | `python -u -m app.handler` | `workersMin >= 1` | 10 min |
| `forge-applio-training` | `python -u -m app.train_handler` | `workersMin = 0` | 6 hours |

## 1. Endpoints

| Use | URL | Notes |
|---|---|---|
| Inference (sync) | `https://api.runpod.ai/v2/<inf-endpoint>/runsync` | 5-minute hard cap |
| Inference (async) | `https://api.runpod.ai/v2/<inf-endpoint>/run` | returns `{id}`, poll `/status/{id}` |
| Training (async) | `https://api.runpod.ai/v2/<train-endpoint>/run` | always async (5–30 min) |
| Job status | `https://api.runpod.ai/v2/<endpoint>/status/{job_id}` | progress included for training |

Endpoint ids are provisioned out-of-band and passed to `voice_service` via
`APPLIO_INFERENCE_ENDPOINT_ID` / `APPLIO_TRAINING_ENDPOINT_ID`.

## 2. Common conventions

### Auth

```
Authorization: Bearer <RUNPOD_API_KEY>
Content-Type: application/json
```

### Request body

Wrap in `{"input": {...}}` and include `action` for inference. Training does
not need `action` (the train handler only supports `train`).

### Response body

```json
{
  "status": "COMPLETED" | "FAILED" | "IN_QUEUE" | "IN_PROGRESS",
  "output": { ... },
  "error": "... (on FAILED)"
}
```

Always check **both** `status == "COMPLETED"` and `output.error`.

## 3. Inference — `action: "convert"`

```json
{
  "input": {
    "action": "convert",
    "input_audio": "s3://bucket/voice/source/input.wav",
    "pth_path": "s3://bucket/voice/model_registry/Applio/logs/<model>/<model>_<n>_best_epoch.pth",
    "index_path": "s3://bucket/voice/model_registry/Applio/logs/<model>/<model>.index",
    "output_s3": "s3://bucket/voice/converted/<job>/output.wav",
    "pitch": 0,
    "index_rate": 0.75,
    "volume_envelope": 1.0,
    "protect": 0.33,
    "f0_method": "rmvpe",
    "embedder_model": "contentvec",
    "export_format": "WAV",
    "clean_audio": false,
    "clean_strength": 0.7,
    "split_audio": false,
    "f0_autotune": false,
    "f0_autotune_strength": 1.0,
    "proposed_pitch": false,
    "proposed_pitch_threshold": 155.0
  }
}
```

Response:

```json
{
  "output": {
    "message": "File /tmp/.../input.wav inferred successfully.",
    "output_audio_uri": "s3://.../converted/<job>/output.wav"
  }
}
```

Set `return_base64: true` instead of (or alongside) `output_s3` to receive
the wav inline:

```json
{
  "output": {
    "message": "...",
    "audio_base64": "...",
    "format": "wav"
  }
}
```

## 4. Training (`/run`, always async)

### 4.1 Start

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

### 4.2 Poll

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

### 4.3 Callback

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

## 5. Environment variables (handlers read these)

| Key | Default | Purpose |
|---|---|---|
| `APPLIO_BASE_DIR` | `/app/Applio` | Source repo root |
| `APPLIO_VOLUME_PATH` | `/runpod-volume/applio` | Network Volume mount |
| `APPLIO_WORK_ROOT` | `/tmp/work` | Ephemeral working root |
| `APPLIO_MODEL_REGISTRY_S3` | `s3://shiftup-enterprise-ai-service/voice/model_registry/Applio` | Training upload destination |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION` | — | S3 access |

## 6. Cold start

First worker spinup takes 60–120s (image pull + torch load). Keep
`workersMin >= 1` on the inference endpoint. Training always runs with
`workersMin = 0` because cold start is negligible compared to a full RVC
run.

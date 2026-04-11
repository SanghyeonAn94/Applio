"""RunPod Serverless training handler for Applio.

Runs the full RVC training pipeline end-to-end:

    raw wavs (S3)
      -> core.run_preprocess_script (slicing / effects / normalization)
      -> core.run_extract_script    (F0 + embedder features)
      -> core.run_train_script      (RVC training, calls run_index_script at the end)
      -> upload {model_name}_best_epoch.pth + {model_name}.index to S3
      -> POST callback

Progress is pushed via ``runpod.serverless.progress_update`` every 5 seconds
so the client can poll ``/status`` for a live stage read.
"""
import glob
import logging
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional

import httpx
import runpod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APPLIO_BASE_DIR = os.environ.get("APPLIO_BASE_DIR", "/app/Applio")
APPLIO_VOLUME_PATH = os.environ.get("APPLIO_VOLUME_PATH", "/runpod-volume/applio")
WORK_ROOT = os.environ.get("APPLIO_WORK_ROOT", "/tmp/work")
MODEL_REGISTRY_S3 = os.environ.get(
    "APPLIO_MODEL_REGISTRY_S3",
    "s3://shiftup-enterprise-ai-service/voice/model_registry/Applio",
)
os.makedirs(WORK_ROOT, exist_ok=True)

os.chdir(APPLIO_BASE_DIR)
sys.path.insert(0, APPLIO_BASE_DIR)

from app import s3_utils
from core import run_preprocess_script, run_extract_script, run_train_script

_PROGRESS_LOCK = threading.Lock()
_progress_state: Dict[str, Any] = {"stage": "init", "step": 0, "total_steps": 0, "extra": {}}


def _set_progress(**fields: Any) -> None:
    with _PROGRESS_LOCK:
        _progress_state.update(fields)


def _snapshot_progress() -> Dict[str, Any]:
    with _PROGRESS_LOCK:
        return dict(_progress_state)


def _find_best_checkpoint(logs_model_dir: str, model_name: str) -> Optional[str]:
    """Return the newest ``*best_epoch.pth`` file for a trained model."""
    matches = sorted(
        glob.glob(os.path.join(logs_model_dir, f"{model_name}_*best_epoch.pth")),
        key=os.path.getmtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _send_callback(url: str, payload: Dict[str, Any]) -> None:
    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(url, json=payload)
            logger.info(f"[Applio Train] callback sent: {response.status_code}")
    except Exception as exc:
        logger.error(f"[Applio Train] callback failed: {exc}")


def _train(event: Dict[str, Any], inp: Dict[str, Any]) -> Dict[str, Any]:
    """Run the complete Applio RVC training pipeline."""
    job_id = uuid.uuid4().hex[:8]
    character_id = inp.get("character_id") or ""
    character_name = inp.get("character_name") or ""
    model_name = inp.get("model_name") or inp.get("exp_name") or character_name or f"job_{job_id}"
    audio_s3 = inp.get("audio_s3_prefix")
    callback_url = inp.get("callback_url")
    training_config = inp.get("training_config") or {}
    if not audio_s3:
        return {"error": "audio_s3_prefix is required"}

    sample_rate = int(training_config.get("sample_rate", 40000))
    total_epoch = int(training_config.get("total_epoch", 100))
    batch_size = int(training_config.get("batch_size", 4))
    f0_method = training_config.get("f0_method", "rmvpe")
    embedder_model = training_config.get("embedder_model", "contentvec")
    vocoder = training_config.get("vocoder", "HiFi-GAN")

    raw_dir = os.path.join(WORK_ROOT, f"applio_raw_{job_id}")
    logs_model_dir = os.path.join(APPLIO_BASE_DIR, "logs", model_name)

    if os.path.exists(logs_model_dir):
        shutil.rmtree(logs_model_dir, ignore_errors=True)

    progress_stop = threading.Event()

    def _push_progress() -> None:
        while not progress_stop.is_set():
            try:
                runpod.serverless.progress_update(event, _snapshot_progress())
            except Exception:
                pass
            time.sleep(5)

    pusher = threading.Thread(target=_push_progress, daemon=True)
    pusher.start()

    result: Dict[str, Any] = {
        "job_id": job_id,
        "character_id": character_id,
        "character_name": character_name,
        "model_name": model_name,
        "status": "failed",
        "error_message": None,
    }

    try:
        _set_progress(stage="download_data", step=0)
        count = s3_utils.download_prefix(audio_s3, raw_dir)
        if count == 0:
            raise RuntimeError(f"no files downloaded from {audio_s3}")

        _set_progress(stage="preprocess", step=0)
        run_preprocess_script(
            model_name=model_name,
            dataset_path=raw_dir,
            sample_rate=sample_rate,
            cpu_cores=int(training_config.get("cpu_cores", 8)),
            cut_preprocess=training_config.get("cut_preprocess", "Automatic"),
            process_effects=bool(training_config.get("process_effects", False)),
            noise_reduction=bool(training_config.get("noise_reduction", True)),
            clean_strength=float(training_config.get("noise_reduction_strength", 0.7)),
            chunk_len=float(training_config.get("chunk_len", 3.0)),
            overlap_len=float(training_config.get("overlap_len", 0.3)),
            normalization_mode=training_config.get("normalization_mode", "none"),
        )

        _set_progress(stage="extract", step=0)
        run_extract_script(
            model_name=model_name,
            f0_method=f0_method,
            cpu_cores=int(training_config.get("cpu_cores", 8)),
            gpu=int(training_config.get("gpu", 0)),
            sample_rate=sample_rate,
            embedder_model=embedder_model,
            embedder_model_custom=training_config.get("embedder_model_custom"),
            include_mutes=int(training_config.get("include_mutes", 2)),
        )

        _set_progress(stage="train", step=0)
        run_train_script(
            model_name=model_name,
            save_every_epoch=int(training_config.get("save_every_epoch", 5)),
            save_only_latest=bool(training_config.get("save_only_latest", False)),
            save_every_weights=bool(training_config.get("save_every_weights", True)),
            total_epoch=total_epoch,
            sample_rate=sample_rate,
            batch_size=batch_size,
            gpu=int(training_config.get("gpu", 0)),
            overtraining_detector=bool(training_config.get("overtraining_detector", True)),
            overtraining_threshold=int(training_config.get("overtraining_threshold", 50)),
            pretrained=bool(training_config.get("pretrained", True)),
            cleanup=bool(training_config.get("cleanup", False)),
            index_algorithm=training_config.get("index_algorithm", "Auto"),
            cache_data_in_gpu=bool(training_config.get("cache_data_in_gpu", False)),
            custom_pretrained=bool(training_config.get("custom_pretrained", False)),
            g_pretrained_path=training_config.get("g_pretrained_path"),
            d_pretrained_path=training_config.get("d_pretrained_path"),
            vocoder=vocoder,
            checkpointing=bool(training_config.get("checkpointing", False)),
        )

        _set_progress(stage="upload", step=0)
        best_model = _find_best_checkpoint(logs_model_dir, model_name)
        if not best_model:
            raise RuntimeError(f"no best_epoch.pth produced in {logs_model_dir}")
        index_file = os.path.join(logs_model_dir, f"{model_name}.index")

        registry_prefix = MODEL_REGISTRY_S3.rstrip("/")
        model_s3 = f"{registry_prefix}/logs/{model_name}/{os.path.basename(best_model)}"
        s3_utils.upload_file(best_model, model_s3, content_type="application/octet-stream")
        result["rvc_model_path"] = model_s3

        if os.path.exists(index_file):
            index_s3 = f"{registry_prefix}/logs/{model_name}/{os.path.basename(index_file)}"
            s3_utils.upload_file(index_file, index_s3, content_type="application/octet-stream")
            result["rvc_index_path"] = index_s3

        result["status"] = "completed"
        _set_progress(stage="completed", step=0)
    except Exception as exc:
        logger.exception("[Applio Train] pipeline failed")
        result["status"] = "failed"
        result["error_message"] = str(exc)
        result["traceback"] = traceback.format_exc()
        _set_progress(stage="failed", step=0)
    finally:
        progress_stop.set()
        pusher.join(timeout=2)
        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(logs_model_dir, ignore_errors=True)

    if callback_url:
        _send_callback(
            callback_url,
            {
                "job_id": job_id,
                "character_id": character_id,
                "character_name": character_name,
                "engine": "applio",
                "status": "completed" if result["status"] == "completed" else "failed",
                "result": {
                    "rvc_model_path": result.get("rvc_model_path"),
                    "rvc_index_path": result.get("rvc_index_path"),
                }
                if result["status"] == "completed"
                else None,
                "error_message": result.get("error_message"),
            },
        )

    return result


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    inp = event.get("input") or {}
    action = inp.get("action") or "train"
    if action != "train":
        return {"error": f"train_handler only supports action='train', got {action!r}"}
    return _train(event, inp)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

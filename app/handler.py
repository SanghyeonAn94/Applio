"""RunPod Serverless inference handler for Applio.

Exposes a single ``convert`` action. The payload carries S3 URIs for the
input audio, the RVC model (``pth_path``), and its FAISS index
(``index_path``); the worker downloads everything, invokes
``core.run_infer_script``, uploads the converted wav, and returns its URI
(or a base64 blob when the caller prefers).
"""
import base64
import logging
import os
import pathlib
import shutil
import sys
import tempfile
import traceback
import uuid
from typing import Any, Dict

import runpod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[1])

APPLIO_BASE_DIR = os.environ.get("APPLIO_BASE_DIR", _REPO_ROOT)
APPLIO_VOLUME_PATH = os.environ.get("APPLIO_VOLUME_PATH", "/runpod-volume/applio")
WORK_ROOT = os.environ.get("APPLIO_WORK_ROOT", "/tmp/work")
os.makedirs(WORK_ROOT, exist_ok=True)

os.chdir(APPLIO_BASE_DIR)
sys.path.insert(0, APPLIO_BASE_DIR)

from app import s3_utils
from core import run_infer_script


def _download_artifact(uri: str, suffix: str) -> str:
    return s3_utils.download_to_temp(uri, suffix=suffix)


def _action_convert(payload: Dict[str, Any]) -> Dict[str, Any]:
    input_uri = payload.get("input_audio")
    pth_uri = payload.get("pth_path")
    if not input_uri or not pth_uri:
        return {"error": "input_audio and pth_path are required"}

    index_uri = payload.get("index_path") or ""
    export_format = payload.get("export_format", "WAV")
    return_base64 = bool(payload.get("return_base64", False))
    output_s3 = payload.get("output_s3")
    if not return_base64 and not output_s3:
        return {"error": "either output_s3 or return_base64=true must be provided"}

    job = uuid.uuid4().hex[:8]
    work_dir = os.path.join(WORK_ROOT, f"convert_{job}")
    os.makedirs(work_dir, exist_ok=True)

    local_input = os.path.join(work_dir, "input" + os.path.splitext(input_uri)[1])
    s3_utils.download_file(input_uri, local_input)

    local_pth = _download_artifact(pth_uri, ".pth")
    local_index = ""
    if index_uri:
        local_index = _download_artifact(index_uri, ".index")

    output_path = os.path.join(work_dir, f"output.{export_format.lower()}")

    try:
        message, final_output = run_infer_script(
            pitch=int(payload.get("pitch", 0)),
            index_rate=float(payload.get("index_rate", 0.75)),
            volume_envelope=float(payload.get("volume_envelope", 1.0)),
            protect=float(payload.get("protect", 0.33)),
            f0_method=payload.get("f0_method", "rmvpe"),
            input_path=local_input,
            output_path=output_path,
            pth_path=local_pth,
            index_path=local_index,
            split_audio=bool(payload.get("split_audio", False)),
            f0_autotune=bool(payload.get("f0_autotune", False)),
            f0_autotune_strength=float(payload.get("f0_autotune_strength", 1.0)),
            proposed_pitch=bool(payload.get("proposed_pitch", False)),
            proposed_pitch_threshold=float(payload.get("proposed_pitch_threshold", 155.0)),
            clean_audio=bool(payload.get("clean_audio", False)),
            clean_strength=float(payload.get("clean_strength", 0.7)),
            export_format=export_format,
            embedder_model=payload.get("embedder_model", "contentvec"),
            embedder_model_custom=payload.get("embedder_model_custom"),
            post_process=bool(payload.get("post_process", False)),
        )

        if not final_output or not os.path.exists(final_output):
            return {"error": f"inference did not produce output: {message}"}

        result: Dict[str, Any] = {"message": message}
        if output_s3:
            s3_utils.upload_file(
                final_output,
                output_s3,
                content_type=f"audio/{export_format.lower()}",
            )
            result["output_audio_uri"] = output_s3
        if return_base64:
            with open(final_output, "rb") as handle:
                result["audio_base64"] = base64.b64encode(handle.read()).decode("ascii")
            result["format"] = export_format.lower()
        return result
    finally:
        for path in (local_pth, local_index):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass
        shutil.rmtree(work_dir, ignore_errors=True)


_ACTIONS = {"convert": _action_convert}


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    inp = event.get("input") or {}
    action = inp.get("action")
    if action not in _ACTIONS:
        return {"error": f"unknown action: {action!r}; valid: {sorted(_ACTIONS)}"}
    try:
        return _ACTIONS[action](inp)
    except Exception as exc:
        logger.exception("[Applio] handler failed")
        return {"error": str(exc), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

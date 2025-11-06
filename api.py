"""
Applio FastAPI Server for Voice Conversion

Similar to GPT-SoVITS api_v2.py structure, provides RESTful endpoints for:
- Voice conversion inference
- Model weight switching
- Health checks

Usage:
    python api.py -a 127.0.0.1 -p 6969

Parameters:
    -a: Bind address (default: 127.0.0.1)
    -p: Port (default: 6969)

API Endpoints:
    POST /infer - Voice conversion inference
    GET /set_model - Switch RVC model weights
    GET /health - Health check
    GET /control - Control commands (restart/exit)
"""

import os
import sys
import argparse
import traceback
import uuid
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime

# Add current directory to path
now_dir = os.getcwd()
sys.path.append(now_dir)

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

# Import Applio core functions
from core import (
    run_infer_script,
    run_preprocess_script,
    run_extract_script,
    run_train_script,
    run_index_script,
)

# Initialize FastAPI app
app = FastAPI(
    title="Applio RVC API",
    description="RESTful API for Applio RVC voice conversion",
    version="1.0.0"
)

# Global state
current_model_path: Optional[str] = None
current_index_path: Optional[str] = None


# Pydantic models for request/response
class InferenceRequest(BaseModel):
    """Voice conversion inference request"""
    input_path: str = Field(..., description="Path to input audio file")
    output_path: str = Field(..., description="Path to output audio file")
    pth_path: str = Field(..., description="Path to RVC model (.pth)")
    index_path: str = Field("", description="Path to index file (.index)")

    # F0 (pitch) parameters
    f0_method: str = Field("rmvpe", description="F0 extraction method (rmvpe, crepe, harvest, etc.)")
    pitch: int = Field(0, description="Pitch shift in semitones")

    # Conversion parameters
    index_rate: float = Field(0.75, description="Feature index influence (0.0-1.0)", ge=0.0, le=1.0)
    volume_envelope: float = Field(1.0, description="Volume envelope mix rate", ge=0.0, le=1.0)
    protect: float = Field(0.33, description="Protect voiceless consonants", ge=0.0, le=0.5)

    # Audio processing
    split_audio: bool = Field(False, description="Split audio into chunks")
    f0_autotune: bool = Field(False, description="Enable auto-tune")
    f0_autotune_strength: float = Field(1.0, description="Auto-tune strength", ge=0.0, le=1.0)

    # Output format
    export_format: str = Field("wav", description="Output format (wav, flac, mp3, etc.)")

    # Advanced parameters
    embedder_model: str = Field("contentvec", description="Speaker embedder model")
    clean_audio: bool = Field(False, description="Clean audio artifacts")
    clean_strength: float = Field(0.7, description="Cleaning strength", ge=0.0, le=1.0)


class InferenceResponse(BaseModel):
    """Voice conversion inference response"""
    success: bool
    output_path: Optional[str] = None
    processing_time_ms: int
    error_message: Optional[str] = None


class SetModelRequest(BaseModel):
    """Model switching request"""
    model_path: str = Field(..., description="Path to RVC model (.pth)")
    index_path: Optional[str] = Field(None, description="Path to index file (.index)")


class ControlRequest(BaseModel):
    """Control command request"""
    command: str = Field(..., description="Control command (restart, exit)")

class PreprocessRequest(BaseModel):
    """Preprocess request for RVC training stage 1"""
    model_name: str = Field(..., description="Model/character name")
    dataset_path: str = Field(..., description="Path to audio files directory")
    sample_rate: int = Field(48000, description="Sample rate (32000, 40000, 48000)")
    cpu_cores: int = Field(8, description="Number of CPU cores to use")
    cut_preprocess: str = Field("Automatic", description="Slicing mode (Skip, Simple, Automatic)")
    process_effects: bool = Field(False, description="Apply high-pass filter")
    noise_reduction: bool = Field(False, description="Enable noise reduction")
    clean_strength: float = Field(0.7, description="Noise reduction strength", ge=0.0, le=1.0)
    chunk_len: float = Field(3.0, description="Chunk length for Simple mode (seconds)")
    overlap_len: float = Field(0.3, description="Overlap length for Simple mode (seconds)")
    normalization_mode: str = Field("none", description="Normalization mode (none, pre, post)")


class ExtractRequest(BaseModel):
    """Extract features request for RVC training stage 2"""
    model_name: str = Field(..., description="Model/character name")
    f0_method: str = Field("rmvpe", description="F0 extraction method")
    cpu_cores: int = Field(8, description="Number of CPU cores")
    gpu: str = Field("0", description="GPU IDs separated by '-' (e.g., '0-1-2'), or '-' for CPU")
    sample_rate: int = Field(48000, description="Sample rate")
    embedder_model: str = Field("contentvec", description="Speaker embedder model")
    embedder_model_custom: str = Field(None, description="Path to custom embedder")
    include_mutes: int = Field(2, description="Number of silent files to include (0-10)")


class TrainRequest(BaseModel):
    """Train RVC model request for RVC training stage 3"""
    model_name: str = Field(..., description="Model/character name")
    save_every_epoch: int = Field(10, description="Save checkpoint every N epochs")
    save_only_latest: bool = Field(False, description="Keep only latest checkpoint")
    save_every_weights: bool = Field(True, description="Save weights every epoch")
    total_epoch: int = Field(1000, description="Total training epochs (1-10000)")
    sample_rate: int = Field(48000, description="Sample rate")
    batch_size: int = Field(8, description="Batch size (1-50)")
    gpu: str = Field("0", description="GPU IDs separated by '-'")
    pretrained: bool = Field(True, description="Use pretrained models")
    custom_pretrained: bool = Field(False, description="Use custom pretrained models")
    g_pretrained_path: str = Field(None, description="Path to custom generator pretrained")
    d_pretrained_path: str = Field(None, description="Path to custom discriminator pretrained")
    overtraining_detector: bool = Field(False, description="Enable overtraining detection")
    overtraining_threshold: int = Field(50, description="Overtraining threshold (1-100)")
    cleanup: bool = Field(False, description="Cleanup previous training files")
    cache_data_in_gpu: bool = Field(False, description="Cache training data in GPU")
    vocoder: str = Field("HiFi-GAN", description="Vocoder type (HiFi-GAN, MRF HiFi-GAN, RefineGAN)")
    checkpointing: bool = Field(False, description="Enable gradient checkpointing")
    index_algorithm: str = Field("Auto", description="Index algorithm (Auto, Faiss, KMeans)")


class IndexRequest(BaseModel):
    """Generate index request for RVC training stage 4"""
    model_name: str = Field(..., description="Model/character name")
    index_algorithm: str = Field("Auto", description="Index algorithm (Auto, Faiss, KMeans)")


class TrainingResponse(BaseModel):
    """Training operation response"""
    success: bool
    message: str
    processing_time_ms: int
    error_message: Optional[str] = None

@app.get("/health")
async def health_check():
    """
    Health check endpoint

    Returns:
        200: Service is healthy
    """
    return {
        "status": "healthy",
        "service": "applio-rvc",
        "version": "1.0.0",
        "current_model": current_model_path,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/set_model")
async def set_model(model_path: str, index_path: Optional[str] = None):
    """
    Switch RVC model weights

    GET:
        http://127.0.0.1:6969/set_model?model_path=/path/to/model.pth&index_path=/path/to/index.index

    Args:
        model_path: Path to RVC model file (.pth)
        index_path: Path to index file (.index) (optional)

    Returns:
        200: Model loaded successfully
        400: Model loading failed
    """
    global current_model_path, current_index_path

    try:
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=400,
                detail=f"Model file not found: {model_path}"
            )

        if index_path and not os.path.exists(index_path):
            raise HTTPException(
                status_code=400,
                detail=f"Index file not found: {index_path}"
            )
        current_model_path = model_path
        current_index_path = index_path or ""

        return {
            "status": "success",
            "model_path": current_model_path,
            "index_path": current_index_path
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to set model: {str(e)}"
        )


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """
    Voice conversion inference

    POST:
    ```json
    {
        "input_path": "/path/to/input.wav",
        "output_path": "/path/to/output.wav",
        "pth_path": "/path/to/model.pth",
        "index_path": "/path/to/index.index",
        "f0_method": "rmvpe",
        "pitch": 0,
        "index_rate": 0.75,
        "volume_envelope": 1.0,
        "protect": 0.33,
        "export_format": "wav"
    }
    ```

    Returns:
        200: Inference successful with output path
        400: Inference failed with error message
    """
    start_time = datetime.now()

    try:
        # Validate input file exists
        if not os.path.exists(request.input_path):
            return InferenceResponse(
                success=False,
                processing_time_ms=0,
                error_message=f"Input file not found: {request.input_path}"
            )

        # Validate model file exists
        if not os.path.exists(request.pth_path):
            return InferenceResponse(
                success=False,
                processing_time_ms=0,
                error_message=f"Model file not found: {request.pth_path}"
            )

        output_dir = os.path.dirname(request.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        info = run_infer_script(
            pitch=request.pitch,
            index_rate=request.index_rate,
            volume_envelope=request.volume_envelope,
            protect=request.protect,
            f0_method=request.f0_method,
            input_path=request.input_path,
            output_path=request.output_path,
            pth_path=request.pth_path,
            index_path=request.index_path,
            split_audio=request.split_audio,
            f0_autotune=request.f0_autotune,
            f0_autotune_strength=request.f0_autotune_strength,
            proposed_pitch=False,
            proposed_pitch_threshold=0.0,
            clean_audio=request.clean_audio,
            clean_strength=request.clean_strength,
            export_format=request.export_format,
            embedder_model=request.embedder_model,
            embedder_model_custom=None,
            formant_shifting=False,
            formant_qfrency=1.0,
            formant_timbre=1.0,
            post_process=False,
            reverb=False,
            pitch_shift=False,
            limiter=False,
            gain=False,
            distortion=False,
            chorus=False,
            bitcrush=False,
            clipping=False,
            compressor=False,
            delay=False,
            reverb_room_size=0.5,
            reverb_damping=0.5,
            reverb_wet_gain=0.5,
            reverb_dry_gain=0.5,
            reverb_width=0.5,
            reverb_freeze_mode=0.5,
            pitch_shift_semitones=0.0,
            limiter_threshold=-6,
            limiter_release_time=0.01,
            gain_db=0.0,
            distortion_gain=25,
            chorus_rate=1.0,
            chorus_depth=0.25,
            chorus_center_delay=7,
            chorus_feedback=0.0,
            chorus_mix=0.5,
            bitcrush_bit_depth=8,
            clipping_threshold=-6,
            compressor_threshold=-20,
            compressor_ratio=4,
            compressor_attack=5.0,
            compressor_release=50.0,
            delay_seconds=0.5,
            delay_feedback=0.5,
            delay_mix=0.5,
        )

        if not os.path.exists(request.output_path):
            return InferenceResponse(
                success=False,
                processing_time_ms=0,
                error_message=f"Inference failed: output file not created. Info: {info}"
            )

        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

        return InferenceResponse(
            success=True,
            output_path=request.output_path,
            processing_time_ms=processing_time
        )

    except Exception as e:
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        error_trace = traceback.format_exc()

        return InferenceResponse(
            success=False,
            processing_time_ms=processing_time,
            error_message=f"Inference error: {str(e)}\n{error_trace}"
        )


@app.post("/control")
async def control(request: ControlRequest):
    """
    Control commands

    POST:
    ```json
    {
        "command": "restart"
    }
    ```

    Commands:
        - restart: Reload the service
        - exit: Shutdown the service

    Returns:
        200: Command executed
    """
    if request.command == "restart":
        return {"status": "restarting"}

    elif request.command == "exit":
        asyncio.create_task(shutdown())
        return {"status": "shutting down"}

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown command: {request.command}"
        )


@app.post("/training/preprocess", response_model=TrainingResponse)
async def preprocess_dataset(request: PreprocessRequest):
    """
    Preprocess audio dataset for RVC training (Stage 1)

    Slices audio files, applies effects, and prepares them for feature extraction.

    POST:
    ```json
    {
        "model_name": "character_name",
        "dataset_path": "/path/to/audio/files",
        "sample_rate": 48000,
        "cpu_cores": 8,
        "cut_preprocess": "Automatic",
        "process_effects": false,
        "noise_reduction": false,
        "clean_strength": 0.7,
        "chunk_len": 3.0,
        "overlap_len": 0.3,
        "normalization_mode": "none"
    }
    ```

    Returns:
        200: Preprocessing successful
        400: Preprocessing failed
    """
    start_time = datetime.now()

    try:
        if not os.path.exists(request.dataset_path):
            return TrainingResponse(
                success=False,
                message="",
                processing_time_ms=0,
                error_message=f"Dataset path not found: {request.dataset_path}"
            )

        message = run_preprocess_script(
            model_name=request.model_name,
            dataset_path=request.dataset_path,
            sample_rate=request.sample_rate,
            cpu_cores=request.cpu_cores,
            cut_preprocess=request.cut_preprocess,
            process_effects=request.process_effects,
            noise_reduction=request.noise_reduction,
            clean_strength=request.clean_strength,
            chunk_len=request.chunk_len,
            overlap_len=request.overlap_len,
            normalization_mode=request.normalization_mode,
        )

        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

        return TrainingResponse(
            success=True,
            message=message,
            processing_time_ms=processing_time
        )

    except Exception as e:
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        error_trace = traceback.format_exc()

        return TrainingResponse(
            success=False,
            message="",
            processing_time_ms=processing_time,
            error_message=f"Preprocessing error: {str(e)}\n{error_trace}"
        )


@app.post("/training/extract", response_model=TrainingResponse)
async def extract_features(request: ExtractRequest):
    """
    Extract features for RVC training (Stage 2)

    Extracts F0 pitch features and speaker embeddings from preprocessed audio.

    POST:
    ```json
    {
        "model_name": "character_name",
        "f0_method": "rmvpe",
        "cpu_cores": 8,
        "gpu": "0",
        "sample_rate": 48000,
        "embedder_model": "contentvec",
        "embedder_model_custom": null,
        "include_mutes": 2
    }
    ```

    Returns:
        200: Feature extraction successful
        400: Feature extraction failed
    """
    start_time = datetime.now()

    try:
        message = run_extract_script(
            model_name=request.model_name,
            f0_method=request.f0_method,
            cpu_cores=request.cpu_cores,
            gpu=request.gpu,
            sample_rate=request.sample_rate,
            embedder_model=request.embedder_model,
            embedder_model_custom=request.embedder_model_custom,
            include_mutes=request.include_mutes,
        )

        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

        return TrainingResponse(
            success=True,
            message=message,
            processing_time_ms=processing_time
        )

    except Exception as e:
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        error_trace = traceback.format_exc()

        return TrainingResponse(
            success=False,
            message="",
            processing_time_ms=processing_time,
            error_message=f"Feature extraction error: {str(e)}\n{error_trace}"
        )


@app.post("/training/train", response_model=TrainingResponse)
async def train_model(request: TrainRequest):
    """
    Train RVC model (Stage 3)

    Trains the RVC model using extracted features. This is the longest operation.

    POST:
    ```json
    {
        "model_name": "character_name",
        "save_every_epoch": 10,
        "save_only_latest": false,
        "save_every_weights": true,
        "total_epoch": 1000,
        "sample_rate": 48000,
        "batch_size": 8,
        "gpu": "0",
        "pretrained": true,
        "custom_pretrained": false,
        "overtraining_detector": false,
        "overtraining_threshold": 50,
        "cleanup": false,
        "cache_data_in_gpu": false,
        "vocoder": "HiFi-GAN",
        "checkpointing": false,
        "index_algorithm": "Auto"
    }
    ```

    Returns:
        200: Training successful (includes index generation)
        400: Training failed
    """
    start_time = datetime.now()

    try:
        message = run_train_script(
            model_name=request.model_name,
            save_every_epoch=request.save_every_epoch,
            save_only_latest=request.save_only_latest,
            save_every_weights=request.save_every_weights,
            total_epoch=request.total_epoch,
            sample_rate=request.sample_rate,
            batch_size=request.batch_size,
            gpu=request.gpu,
            overtraining_detector=request.overtraining_detector,
            overtraining_threshold=request.overtraining_threshold,
            pretrained=request.pretrained,
            cleanup=request.cleanup,
            index_algorithm=request.index_algorithm,
            cache_data_in_gpu=request.cache_data_in_gpu,
            custom_pretrained=request.custom_pretrained,
            g_pretrained_path=request.g_pretrained_path,
            d_pretrained_path=request.d_pretrained_path,
            vocoder=request.vocoder,
            checkpointing=request.checkpointing,
        )

        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

        return TrainingResponse(
            success=True,
            message=message,
            processing_time_ms=processing_time
        )

    except Exception as e:
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        error_trace = traceback.format_exc()

        return TrainingResponse(
            success=False,
            message="",
            processing_time_ms=processing_time,
            error_message=f"Training error: {str(e)}\n{error_trace}"
        )


@app.post("/training/index", response_model=TrainingResponse)
async def generate_index(request: IndexRequest):
    """
    Generate index file for RVC model (Stage 4)

    Creates FAISS index for feature retrieval during inference.
    Usually called automatically by training, but can be called separately.

    POST:
    ```json
    {
        "model_name": "character_name",
        "index_algorithm": "Auto"
    }
    ```

    Returns:
        200: Index generation successful
        400: Index generation failed
    """
    start_time = datetime.now()

    try:
        message = run_index_script(
            model_name=request.model_name,
            index_algorithm=request.index_algorithm
        )

        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

        return TrainingResponse(
            success=True,
            message=message,
            processing_time_ms=processing_time
        )

    except Exception as e:
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        error_trace = traceback.format_exc()

        return TrainingResponse(
            success=False,
            message="",
            processing_time_ms=processing_time,
            error_message=f"Index generation error: {str(e)}\n{error_trace}"
        )


async def shutdown():
    """Graceful shutdown"""
    await asyncio.sleep(1)
    os._exit(0)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Applio RVC API Server"
    )
    parser.add_argument(
        "-a", "--address",
        type=str,
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=6969,
        help="Port (default: 6969)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"""
╔══════════════════════════════════════╗
║      Applio RVC API Server          ║
╚══════════════════════════════════════╝

Address: {args.address}:{args.port}
API Docs: http://{args.address}:{args.port}/docs

Inference Endpoints:
  POST /infer                  - Voice conversion
  GET  /set_model              - Switch model
  GET  /health                 - Health check
  POST /control                - Control commands

Training Endpoints:
  POST /training/preprocess    - Stage 1: Preprocess audio
  POST /training/extract       - Stage 2: Extract features
  POST /training/train         - Stage 3: Train RVC model
  POST /training/index         - Stage 4: Generate index
""")

    uvicorn.run(
        app,
        host=args.address,
        port=args.port,
        log_level="info"
    )

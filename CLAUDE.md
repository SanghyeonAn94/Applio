# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with Applio RVC integration.

## Project Overview

Applio is an RVC (Retrieval-based Voice Conversion) model integrated into SAI2ply's Voice ML Service for voice conversion capabilities. Similar to GPT-SoVITS integration, Applio provides a FastAPI server for voice conversion inference.

## Environment Setup

### Running Applio API Server

**Standalone:**
```bash
python api.py -a 127.0.0.1 -p 6969
```

**Docker:**
```bash
docker-compose -f docker-compose.dev.yml up applio-rvc-dev
```

Parameters:
- `-a`: Bind address (default: 127.0.0.1)
- `-p`: Port (default: 6969)

## API Endpoints

### POST /infer
Voice conversion inference

**Request:**
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

  "split_audio": false,
  "f0_autotune": false,
  "f0_autotune_strength": 1.0,

  "export_format": "wav",
  "embedder_model": "contentvec",
  "clean_audio": false,
  "clean_strength": 0.7
}
```

**Response:**
```json
{
  "success": true,
  "output_path": "/path/to/output.wav",
  "processing_time_ms": 5000,
  "error_message": null
}
```

### GET /set_model
Switch RVC model weights

**Query Parameters:**
- `model_path`: Path to RVC model file (.pth)
- `index_path`: Path to index file (.index) (optional)

**Response:**
```json
{
  "status": "success",
  "model_path": "/path/to/model.pth",
  "index_path": "/path/to/index.index"
}
```

### GET /health
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "service": "applio-rvc",
  "version": "1.0.0",
  "current_model": "/path/to/current/model.pth",
  "timestamp": "2025-11-06T12:00:00"
}
```

### POST /control
Control commands (restart/exit)

**Request:**
```json
{
  "command": "restart"  // or "exit"
}
```

## RVC Parameters Explained

### F0 (Pitch) Parameters
- **f0_method**: F0 extraction method
  - `rmvpe` (default, recommended): Robust pitch extractor
  - `crepe`: Deep learning-based pitch tracker
  - `harvest`: Traditional pitch extractor
  - `dio`: High-speed pitch extractor
  - `pm`: Simplified pitch marker

- **pitch**: Pitch shift in semitones (-12 to +12)
  - Positive: Higher pitch
  - Negative: Lower pitch

### Conversion Parameters
- **index_rate**: Feature index influence (0.0-1.0)
  - Controls how much the index file affects conversion
  - Higher = more similar to training voice

- **volume_envelope**: Volume envelope mix rate (0.0-1.0)
  - Controls dynamic preservation from source audio

- **protect**: Protect voiceless consonants (0.0-0.5)
  - Prevents over-processing of consonants

### Audio Processing
- **split_audio**: Split long audio into chunks for processing
- **f0_autotune**: Enable pitch correction
- **f0_autotune_strength**: Auto-tune intensity (0.0-1.0)

### Advanced Parameters
- **embedder_model**: Speaker embedder model
  - `contentvec` (default): General-purpose embedder
  - Other options available in Applio core

- **clean_audio**: Apply audio artifact cleaning
- **clean_strength**: Cleaning intensity (0.0-1.0)

## Integration Architecture

### Hexagonal Architecture Pattern

Similar to GPT-SoVITS, Applio follows the same hexagonal architecture:

```
Voice ML Service
├── domain/
│   ├── aggregates/
│   │   └── model_registry.py          # Registers Applio models
│   ├── models/
│   │   └── inference.py                # ModelVersionPort interface
│   └── ports/
│       └── ModelVersionPort            # Port interface
│
├── infrastructure/
│   └── adapters/
│       └── applio/
│           ├── __init__.py
│           └── inference_adapter.py    # ApplioCloudAdapter
│
└── application/
    └── services/
        └── voice_conversion_application_service.py
```

### ApplioCloudAdapter

Implements `ModelVersionPort` interface with the following key methods:

**Capabilities:**
- `VOICE_CONVERSION`
- `VOICE_CLONING`

**Key Methods:**
- `can_handle(request)`: Determines if Applio can handle the inference request
  - Returns `True` for voice conversion requests (no text, has ref_audio)

- `execute_inference(request)`: Executes voice conversion
  1. Setup character-specific RVC model
  2. Prepare source audio (download from Data Service if needed)
  3. Build Applio API payload
  4. Make request with retry logic
  5. Upload result to Data Service
  6. Cleanup temporary files

- `health_check()`: Verifies Applio API is healthy

### Model Registration

Applio models are registered in `ModelRegistryAggregate.initialize_default_models()`:

```python
async def _register_default_applio(self) -> None:
    adapter = ApplioCloudAdapter(
        version="v1",
        applio_api_endpoint=settings.applio_api_endpoint,
        api_service_url=settings.api_service_url,
        data_service_url=settings.data_service_url,
        timeout_seconds=60,
        max_retries=3
    )

    self.register_model_version(
        model_id="applio-rvc",
        version="v1",
        adapter=adapter,
        api_endpoint=settings.applio_api_endpoint
    )
```

### Character-Specific Models

Similar to GPT-SoVITS checkpoint loading, Applio supports character-specific RVC models:

1. **Fetch Model Info**: GET `/api/statements/character/{character_id}/rvc-model`
   - Returns: `rvc_model_path`, `index_path`

2. **Load Model**: GET `/set_model?model_path=...&index_path=...`
   - Switches to character-specific RVC model

3. **Caching**: Adapter caches loaded model to avoid redundant switches

## Data Flow

### Voice Conversion Request Flow

```
Frontend (VoiceSwapPage.tsx)
→ Front Service (/api/voices/swaps)
→ Voice ML Service (/voice-conversion/execute)
→ VoiceConversionApplicationService
→ MLDomainService.execute_voice_conversion()
→ ModelRegistry.select_capable_model()
→ ApplioCloudAdapter.execute_inference()
→ Applio API (POST /infer)
→ RVC Inference (core.run_infer_script)
→ Return converted audio
→ Upload to Data Service
→ Return audio URL
```

### Request DTO Structure

**Frontend → Front Service:**
```typescript
{
  characterId: string;
  referenceAudioFile: File | File[];
  voiceId?: string;
  n_samples?: number;
}
```

**Front Service → Voice ML Service:**
```python
VoiceConversionRequestDTO(
    request_id: str,
    reference_audio_path: str,
    character_id: str,
    text_lang: str = "auto"
)
```

**Voice ML Service → Applio Adapter:**
```python
InferenceRequest(
    request_id: str,
    character_id: str,
    ref_audio_path: str,
    metadata: Dict[str, Any]  # RVC parameters
)
```

## Important Implementation Notes

### File Handling

**Temporary Files:**
- Input audio: `shared/cache_storage/temp_audio/{uuid}.wav`
- Output audio: `shared/cache_storage/applio_outputs/{request_id}_converted.wav`
- Always cleaned up in `finally` block

**Model Files:**
- RVC models (.pth): Stored in `/app/logs` (Docker volume)
- Index files (.index): Stored alongside models
- Fetched from API Service based on character_id

### Error Handling

- **Retry Logic**: 3 retries with exponential backoff (2, 4, 6 seconds)
- **Graceful Degradation**: Returns error in `InferenceResult` without raising exceptions
- **Cleanup**: Ensures temp files are cleaned even on error

### Memory Management

- Docker container: 8GB shared memory (`shm_size: '8gb'`)
- GPU: Requires NVIDIA GPU with CUDA support
- Model caching: Only one model loaded at a time per adapter instance

### Performance Considerations

- **Timeout**: 60 seconds default (configurable)
- **Sequential Processing**: For directory uploads, files processed one by one
- **Health Checks**: Regular health checks prevent requests to unhealthy services

## Docker Configuration

### Dockerfile

- Base: `python:3.10-bullseye`
- Port: 6969
- Entry: `python3 api.py -a 0.0.0.0 -p 6969`
- GPU: Requires CUDA 12.8 (PyTorch 2.7.1)

### Docker Compose

```yaml
applio-rvc-dev:
  build:
    context: models/Applio
    dockerfile: Dockerfile
  container_name: sai2ply-applio-rvc-dev
  ports:
    - "6969:6969"
  volumes:
    - ../shared:/app/shared
    - ml_dev_applio_models:/app/logs
  networks:
    - sai2ply-ml
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

## Environment Variables

```bash
# In configs/env.dev
APPLIO_API_URL=http://applio-rvc-dev:6969

# In voice_ml_service/app/settings.py
applio_api_endpoint: str = os.getenv("APPLIO_API_URL", "http://localhost:6969")
```

## Testing

### Health Check
```bash
curl http://localhost:6969/health
```

### Model Loading
```bash
curl "http://localhost:6969/set_model?model_path=/path/to/model.pth&index_path=/path/to/index.index"
```

### Voice Conversion
```bash
curl -X POST http://localhost:6969/infer \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "/app/shared/cache_storage/temp_audio/input.wav",
    "output_path": "/app/shared/cache_storage/applio_outputs/output.wav",
    "pth_path": "/app/logs/model.pth",
    "index_path": "/app/logs/index.index",
    "f0_method": "rmvpe",
    "pitch": 0,
    "index_rate": 0.75
  }'
```

## Common Issues

1. **GPU Not Available**: Ensure Docker has GPU support and NVIDIA drivers installed
2. **Model Not Found**: Check model paths in `/app/logs` volume
3. **Audio Format Issues**: Applio expects WAV format (other formats need conversion)
4. **Timeout Errors**: Increase `timeout_seconds` in adapter initialization for longer audio
5. **Memory Issues**: Increase `shm_size` in Docker Compose if needed

## Comparison with GPT-SoVITS

| Feature | GPT-SoVITS | Applio RVC |
|---------|-----------|------------|
| **Primary Use** | TTS (Text-to-Speech) | Voice Conversion |
| **Input** | Text + Reference Audio | Source Audio + Target Voice Model |
| **Training** | Two-stage (GPT + SoVITS) | Single-stage (RVC) |
| **Model Size** | Larger (~1-2GB) | Smaller (~100-500MB) |
| **Processing Time** | ~5-15s per sentence | ~3-10s per audio file |
| **Language Support** | Multi-language (CN/EN/JP/KR) | Language-agnostic |
| **Port** | 9881 | 6969 |

## Future Enhancements

- [ ] Support for batch processing multiple files in single request
- [ ] Implement RVC model training pipeline
- [ ] Add post-processing effects (reverb, EQ, etc.)
- [ ] Support for more embedder models
- [ ] Optimize model loading/caching strategy

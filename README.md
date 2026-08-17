# Universal 3.5 Pro Sync - Input/Output Specification and Samples

Reference material for the **Universal 3.5 Pro Sync** speech-to-text model package on AWS Marketplace.
This repository contains sample input audio, the exact output each sample produces, a sample notebook,
and runnable examples for real-time and batch inference.

- [`universal-3-5-pro-sync-usage.ipynb`](universal-3-5-pro-sync-usage.ipynb) - end-to-end sample notebook (subscribe, deploy, transcribe, batch, clean up)
- [`data/input/real-time/`](data/input/real-time) - sample audio for real-time inference
- [`data/input/batch/`](data/input/batch) - sample audio for batch transform
- [`data/output/real-time/`](data/output/real-time) - the exact response each real-time sample produces (with `{"timestamps": true}`)
- [`data/output/batch/`](data/output/batch) - the exact batch transform output for each batch input (no configuration, so no timestamps)
- [`examples/`](examples) - runnable Python examples

---

## Deployment

> ### Required: pin the inference AMI version
>
> Set `InferenceAmiVersion="al2-ami-sagemaker-inference-gpu-2-1"` on the endpoint configuration's
> production variant.
>
> That AMI provides NVIDIA driver 535 with CUDA-compat mounting disabled. Without it, the host mounts
> CUDA compatibility libraries that conflict with the container's CUDA runtime and the container exits
> during startup with:
>
> ```
> RuntimeError: Unexpected error from cudaGetDeviceCount().
> Error 803: system has unsupported display driver / cuda driver combination
> ```
>
> The endpoint then fails its ping health check. `InferenceAmiVersion` is a property of the endpoint
> configuration, so it must be set at deploy time.

Endpoint creation takes roughly 7-12 minutes: the container image is large and model weights are loaded
onto the GPU at startup. Set `ContainerStartupHealthCheckTimeoutInSeconds` to at least `1800`.

Marketplace model packages run in network isolation and cannot be deployed to SageMaker Serverless
Inference endpoints.

---

## Input

**Content types:** `audio/wav`, `audio/pcm`

**Recommended format:** 16 kHz, mono, 16-bit PCM WAV.

**Maximum audio duration:** 120 seconds per request. Longer audio is rejected with HTTP 413:

```json
{"status":413,"title":"Audio Too Large","detail":"audio duration 281051 ms exceeds limit 120000 ms"}
```

### Two request forms

**1. Raw audio** - send the audio bytes directly with `Content-Type: audio/wav`. Uses all defaults.

**2. Multipart with configuration** - required to pass any option, including timestamps. Send
`multipart/form-data` with two parts:

| Part | Content-Type | Contents |
|------|--------------|----------|
| `audio` | `audio/wav` | the audio bytes |
| `config` | `application/json` | a JSON object of options |

```
--BOUNDARY
Content-Disposition: form-data; name="audio"; filename="audio.wav"
Content-Type: audio/wav

<raw audio bytes>
--BOUNDARY
Content-Disposition: form-data; name="config"
Content-Type: application/json

{"timestamps": true}
--BOUNDARY--
```

See [`examples/invoke_realtime.py`](examples/invoke_realtime.py) for a working implementation.

### Configuration options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `timestamps` | boolean | `false` | Include word-level `start` / `end` times in the response. |

<!-- TODO(assemblyai): confirm semantics and defaults before publishing. The following fields are
     accepted by the request schema but their behavior is not documented here yet:
     sample_rate (int), channels (int), prompt (str), language_code, conversation_context,
     and one of keyterms / keyterms_prompt / word_boost (mutually exclusive). -->

Unrecognized options are rejected with HTTP 400 naming the offending field, so typos fail loudly rather
than being silently ignored:

```json
{"status":400,"title":"Bad Request","detail":"invalid config part: nonexistent_option: Extra inputs are not permitted"}
```

---

## Output

Content type: `application/json`

```json
{
  "text": "Smoke from hundreds of wildfires in Canada is triggering air quality alerts throughout the US.",
  "words": [
    {"text": "Smoke", "confidence": 0.9957, "start": 32, "end": 449},
    {"text": "from",  "confidence": 0.9985, "start": 674, "end": 819}
  ],
  "confidence": 0.982,
  "audio_duration_ms": 60000,
  "session_id": "ee3a2cd4-69a6-49d5-8968-4095b4bb8589",
  "request_time_ms": 2368.4
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Full transcript, formatted and punctuated. |
| `words[].text` | string | The word as it appears in the transcript. |
| `words[].confidence` | float | Model confidence for that word, 0-1. |
| `words[].start`, `words[].end` | integer | Milliseconds from the start of the audio. Present only when `timestamps` is `true`. Omitted for any word that cannot be aligned - timings are exact or absent, never estimated. |
| `confidence` | float | Mean confidence across the transcript. |
| `audio_duration_ms` | integer | Duration of the submitted audio. |
| `session_id` | string | Unique identifier for the request. |
| `request_time_ms` | float | Server-side processing time. |

---

## Batch transform

Batch transform processes many files from Amazon S3 in one job. Each object is sent as a single request
and its transcript is written to the output prefix as `<filename>.out`.

**Batch transform cannot pass configuration.** Each S3 object is sent as a raw request body, so there is
no way to attach a `config` part. Batch output therefore contains word text and confidence but **not**
timestamps. Use a real-time endpoint when you need word-level timings.

Set `SplitType` to `"None"` so each file is treated as one record rather than being split mid-audio. See
[`examples/invoke_batch.py`](examples/invoke_batch.py).

---

## Service limits

| Limit | Real-time endpoint | Batch transform |
|-------|-------------------|-----------------|
| Maximum audio duration per request | 120 seconds | 120 seconds |
| Maximum payload size | 25 MB | 100 MB |
| Maximum processing time per request | 60 seconds | 60 minutes |

---

## Samples

| Input | Duration | Real-time output (timestamps on) | Batch output (defaults) |
|-------|----------|----------------------------------|-------------------------|
| [`sample-10s.wav`](data/input/real-time/sample-10s.wav) | 10 s | [`sample-10s.json`](data/output/real-time/sample-10s.json) | [`sample-10s.wav.out`](data/output/batch/sample-10s.wav.out) |
| [`sample-30s.wav`](data/input/real-time/sample-30s.wav) | 30 s | [`sample-30s.json`](data/output/real-time/sample-30s.json) | [`sample-30s.wav.out`](data/output/batch/sample-30s.wav.out) |
| [`sample-60s.wav`](data/input/real-time/sample-60s.wav) | 60 s | [`sample-60s.json`](data/output/real-time/sample-60s.json) | [`sample-60s.wav.out`](data/output/batch/sample-60s.wav.out) |

Real-time outputs were produced with `{"timestamps": true}` via a multipart request; batch outputs
are the raw request-body form with default settings, exactly as a batch transform job produces them.
The same three files appear under `data/input/batch/` for use with batch transform.

All outputs in this repository are real captures from the model package running on `ml.g5.xlarge`.

## Sample audio attribution

The sample recordings are excerpts from the LibriVox recording of *Pride and Prejudice* by Jane
Austen (solo version), converted to 16 kHz mono 16-bit PCM WAV. LibriVox recordings are in the
public domain: https://archive.org/details/solo_pride_librivox

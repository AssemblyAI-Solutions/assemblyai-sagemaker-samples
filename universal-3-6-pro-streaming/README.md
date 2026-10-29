# Universal-3.6 Pro Streaming - Input/Output Specification and Samples

Reference material for the **Universal-3.6 Pro Streaming** speech-to-text model package on AWS
Marketplace. One endpoint serves:

- **live streaming** over `InvokeEndpointWithBidirectionalStream`: send audio as it is captured,
  receive partial and final transcripts as the speaker talks;
- **whole-file transcription** over `InvokeEndpoint`, asynchronous inference and batch transform.

This folder contains:

- [`universal-3-6-pro-streaming-usage.ipynb`](universal-3-6-pro-streaming-usage.ipynb) - end-to-end
  sample notebook (subscribe, deploy, stream, transcribe a file, batch, usage metrics, clean up)
- [`examples/`](examples) - runnable Python examples
- [`data/input/`](data/input) - sample audio
- [`data/output/`](data/output) - the exact output each sample produces, captured from the model package
- [`requirements.txt`](requirements.txt) - Python packages for the examples

---

## Quick start

```bash
pip install -r requirements.txt          # Python 3.12+

python examples/deploy.py --model-package-arn <arn> --role-arn <sagemaker-execution-role-arn>
python examples/stream_file.py universal-3-6-pro-streaming data/input/sample-10s.wav
python examples/invoke_file.py universal-3-6-pro-streaming data/input/sample-10s.wav
uv run examples/stream_mic.py universal-3-6-pro-streaming   # live from your microphone

python examples/deploy.py --delete
```

| Example | What it does |
|---------|--------------|
| [`deploy.py`](examples/deploy.py) | Creates the model, endpoint configuration and endpoint; `--delete` removes all three |
| [`stream_file.py`](examples/stream_file.py) | Streams a WAV file in real time and prints partial and final turns |
| [`stream_mic.py`](examples/stream_mic.py) | Live-transcribes your microphone (a self-contained `uv run` script) |
| [`invoke_file.py`](examples/invoke_file.py) | Transcribes a whole file with `InvokeEndpoint` |
| [`invoke_batch.py`](examples/invoke_batch.py) | Transcribes files in S3 with a batch transform job |

---

## Deployment

| | |
|---|---|
| Real-time instance types | `ml.g6e.xlarge`, `ml.g6e.2xlarge`, `ml.g6e.4xlarge` (NVIDIA L40S); `ml.g7e.2xlarge`, `ml.g7e.4xlarge` (NVIDIA RTX PRO 6000); one GPU each |
| Batch transform instance type | `ml.g5.2xlarge` |
| `InferenceAmiVersion` (real-time) | `al2023-ami-sagemaker-inference-gpu-4-1` - recommended |
| `TransformAmiVersion` (batch) | `al2-ami-sagemaker-batch-gpu-535`; pass no `Environment` |
| `ContainerStartupHealthCheckTimeoutInSeconds` | `1800` |
| `ModelDataDownloadTimeoutInSeconds` | `1800` |

The model needs CUDA 13 (NVIDIA driver 580 or later, or 535+ with CUDA compatibility libraries,
which the container enables itself). The `al2023-ami-sagemaker-inference-gpu-4-1` AMI provides
driver 580, so the container runs natively; on an older default driver it falls back to the
compatibility libraries. Check that your account has a quota for the instance type (for example
`ml.g6e.xlarge for endpoint usage`); new accounts often start at 0.

The endpoint is `InService` once the weights are on the GPU and the model has warmed up; until then
`/ping` answers 503. Marketplace model packages run with network isolation and cannot be deployed to
Serverless Inference.

---

## Live streaming

Open a stream with `InvokeEndpointWithBidirectionalStream` on the SageMaker runtime's HTTP/2
endpoint, `https://runtime.sagemaker.<region>.amazonaws.com:8443`. In Python this is AWS's
`aws-sdk-sagemaker-runtime-http2` package (see [`requirements.txt`](requirements.txt) and
[`examples/stream_file.py`](examples/stream_file.py)).

Session options go in the request's `ModelQueryString` as URL query parameters. Every message on the
stream is AssemblyAI's [v3 streaming protocol](https://www.assemblyai.com/docs/api-reference/streaming-api/streaming-api),
so existing AssemblyAI streaming code ports directly.

```
ModelQueryString: sample_rate=16000&encoding=pcm_s16le&keyterms_prompt=["LibriVox"]
```

### Sending

| Payload | `DataType` | Contents |
|---------|-----------|----------|
| Audio | `BINARY` | Audio in the session's `encoding`. PCM: 20-1000 ms per message (50 ms is typical). Send at real time, as it is captured. |
| `{"type": "Terminate"}` | `UTF8` | Flush the last turn and end the session; the server replies with `Termination`. |
| `{"type": "ForceEndpoint"}` | `UTF8` | End the current turn now. |
| `{"type": "KeepAlive"}` | `UTF8` | Resets the `inactivity_timeout` timer; only needed when that parameter is set and no audio is being sent. Without it a session stays open until `Terminate` or the 30-minute stream limit, and the container answers SageMaker's transport pings itself. |
| `{"type": "UpdateConfiguration", ...}` | `UTF8` | Change turn-detection settings, `keyterms_prompt` or `prompt` mid-session. |

### Receiving

Each event's payload is one JSON message ([`data/output/streaming/sample-10s.jsonl`](data/output/streaming/sample-10s.jsonl)
is a complete session):

| `type` | When |
|--------|------|
| `Begin` | The session is open; carries the session `id` and its effective `configuration`. |
| `SpeechStarted` | Speech was detected; carries its `timestamp` (ms into the audio). |
| `Turn` | A transcript update. `end_of_turn: false` is a partial that will change; `end_of_turn: true` is the final, formatted text of that turn. `turn_order` numbers the turns; `words` carries per-word `start`/`end` (ms) and `confidence`. |
| `Termination` | The session ended; carries `audio_duration_seconds`. |
| `Error` | The session was refused or failed; carries `error_code` and `error`. |

When the container closes a stream, SageMaker reports the close as a `ModelStreamError`, which the
Python SDK raises from `receive()`, **including after a normal `Termination`**. Treat the stream as
successful if `Termination` arrived; after an `Error`, the `ModelStreamError` is how the session ends.

Use one SDK client (one HTTP/2 connection) per concurrent stream; streams that share a client can stall.

### Connection parameters

| Parameter | Description |
|-----------|-------------|
| `sample_rate` | Audio sample rate in Hz. Default `16000`. |
| `encoding` | `pcm_s16le` (16-bit little-endian mono PCM, default), `pcm_mulaw`, `opus` (raw Opus, one packet per message), `ogg_opus` (an Ogg Opus stream, in any chunks) or `aac` (an ADTS stream, in any chunks). `sample_rate` applies to the PCM encodings. |
| `speech_model` | Optional; if set, must be `universal-3-6-pro`. |
| `language_codes` | JSON list of up to 10 expected languages, e.g. `["en","es"]`. |
| `language_detection` | `true` to add `language_code` and `language_confidence` to each final turn. |
| `keyterms_prompt` | JSON list of up to 100 words and phrases to recognise, e.g. `["AssemblyAI","SageMaker"]`. |
| `prompt` | Free-text context about the audio (up to 1750 characters). |
| `agent_context` | What a voice agent just said, to improve recognition of the reply (up to 1750 characters). |
| `mode` | `balanced` (default), `min_latency` or `max_accuracy`. |
| `end_of_turn_confidence_threshold` | Turn-detection confidence needed to end a turn on a short silence. |
| `min_end_of_turn_silence_when_confident` | Silence (ms) before ending a confident turn. |
| `max_turn_silence` | Silence (ms) after which a turn always ends. |
| `include_partial_turns` | `false` to receive final turns only. |
| `continuous_partials` | `true` to keep sending partials through long turns; `false` for one early partial per turn. |
| `inactivity_timeout` | Seconds of inactivity (5-3600) before the session ends; off by default. |

Invalid values are refused with the streaming API's usual `Error` message and close code `3006`,
naming the problem. Parameter names that are not recognised are ignored, so check the spelling of
any option that seems to have no effect.

### Not available on this endpoint

The model package contains the speech model only. Options that need other AssemblyAI services are
refused with an `Error` and close code `3006` (HTTP 400 for whole-file requests) rather than silently
ignored:

| Refused | Needs |
|---------|-------|
| `redact_pii` | PII redaction |
| `speaker_labels` | speaker diarization |
| `noise_suppression_model`, `voice_focus` | noise suppression |
| `custom_context`, `llm_gateway` | LLM transcript enhancement |
| `domain` | domain-specific correction |
| `webhook_url` and related | outbound network access (Marketplace containers have none) |
| any other `speech_model` | the endpoint serves only `universal-3-6-pro` |

### Limits

| Limit | Value |
|-------|-------|
| Stream duration | 30 minutes per stream (a SageMaker limit); reconnect to continue |
| Audio per message | 20-1000 ms |
| Concurrent streams per instance | 32 by default (`MAX_OPEN_STREAMS`, see [Configuration](#configuration)) |

Every supported instance has one GPU, so capacity scales with instance count, not instance size:
plan on about 16 concurrent streams per instance for latency-sensitive voice agents and up to 32 for
throughput, and add instances (or SageMaker auto scaling) beyond that.

---

## Whole-file transcription

`POST /invocations`, called through `InvokeEndpoint`, asynchronous inference or batch transform,
transcribes a complete file and returns every turn at once. (Asynchronous inference needs an endpoint
configuration with `AsyncInferenceConfig`; options go in `CustomAttributes` of `InvokeEndpointAsync`.)

**Request body**, one of:

| `ContentType` | Body |
|---------------|------|
| any `audio/*` type, e.g. `audio/wav` | An audio file: WAV, FLAC, OGG or MP3, 8-48 kHz, any channel count (mixed down to mono). The format is detected from the file's contents. |
| `audio/l16` | Raw 16-bit little-endian mono PCM; `sample_rate` must be given in the config. |
| `application/json` | `{"audio": "<base64 of either of the above>", "config": {...}}` |

**Options** are the connection parameters above, as a JSON object. For audio bodies, pass it in the
`CustomAttributes` field of `InvokeEndpoint` (the `X-Amzn-SageMaker-Custom-Attributes` header).
`CustomAttributes` is limited to 1024 characters; send larger options, such as a long `prompt`, in a
JSON body instead:

```python
smr.invoke_endpoint(
    EndpointName="universal-3-6-pro-streaming",
    ContentType="audio/wav",
    Body=open("data/input/sample-10s.wav", "rb").read(),
    CustomAttributes=json.dumps({"keyterms_prompt": ["LibriVox"]}),
)
```

**Response** ([`data/output/invocations/sample-10s.json`](data/output/invocations/sample-10s.json)):

```json
{
  "session_id": "6f10a424-2795-4484-9988-e98bd405127b",
  "text": "This is a LibriVox recording. All LibriVox recordings are in the public domain.",
  "audio_duration_ms": 10000,
  "turns": [
    {
      "turn_order": 0,
      "transcript": "This is a LibriVox recording. All LibriVox recordings are in the public domain.",
      "start_ms": 2240,
      "end_ms": 9882,
      "words": [
        {"start": 2240, "end": 2680, "text": "This", "confidence": 0.9986, "word_is_final": true},
        {"start": 2702, "end": 2922, "text": "is", "confidence": 0.999988, "word_is_final": true}
      ],
      "end_of_turn_confidence": 1.0,
      "turn_is_formatted": true
    }
  ]
}
```

| Field | Description |
|-------|-------------|
| `text` | The full transcript: the final turns joined with spaces. |
| `turns` | Each final turn: its `transcript`, `start_ms` / `end_ms`, and `words` with per-word `start` / `end` (ms) and `confidence`. |
| `audio_duration_ms` | Duration of the submitted audio. |
| `session_id` | Unique identifier for the request. |

Errors are raised by `boto3` as `ModelError`, with the container's HTTP status and message:

| Status | Meaning |
|--------|---------|
| 400 | Invalid or refused options (JSON `{"error": "..."}` naming the problem) |
| 413 | Request body too large |
| 415 | Unreadable or unsupported audio |
| 503 | The model is still loading |
| 504 | The file was not transcribed in time |

**Long files.** `InvokeEndpoint` allows 60 seconds per request, which suits files of about a minute.
Use asynchronous inference or batch transform for longer recordings.

---

## Batch transform

Each S3 object is one record; its result is written to the output prefix as `<filename>.out`, in the
response format above. See [`examples/invoke_batch.py`](examples/invoke_batch.py).

- Set `SplitType` to `"None"` so a file is never split.
- Set `TransformAmiVersion="al2-ami-sagemaker-batch-gpu-535"` and pass no `Environment`: SageMaker
  rejects it on a Marketplace package's job, and the container enables CUDA compatibility itself.
- The job needs the `ml.g5.2xlarge for transform job usage` quota in your account.
- Leave `MaxConcurrentTransforms` and `MaxPayloadInMB` unset to use the container's defaults: 4
  concurrent records of up to 25 MB each.
- A record must finish within `ModelClientConfig.InvocationsTimeoutInSeconds` (default 600, maximum
  3600); raise it for long recordings.
- Batch transform cannot set `CustomAttributes`. To pass options, store each file as a JSON envelope
  (`{"audio": "<base64>", "config": {...}}`) and run the job with `ContentType="application/json"`;
  [`data/input/batch-json/`](data/input/batch-json) has an example.

---

## Billing and usage metrics

The software charge is metered by the amount of audio processed, in units of **8 seconds of audio**
(see the listing for the price per unit). A stream is billed for the duration of the audio sent to it
from when the session begins until `Terminate` (for Opus and AAC, the duration the frames encode),
reported as the stream runs; a whole-file request or batch record is billed for the file's duration. Each stream
and request is rounded up to a whole unit. A request that fails is not billed; a stream that drops is
billed for the audio it received. The AWS infrastructure charge for the instance is separate.

Every stream and request also writes its usage to the container's CloudWatch logs (the endpoint's or
the transform job's) in Embedded Metric
Format, so you can graph, alarm on and reconcile usage in your own account:

| | |
|---|---|
| Namespace | `AssemblyAI/SageMakerStreaming` |
| Metrics | `ConsumedUnits` (billed units), `AudioDurationSeconds` |
| Dimension | `Operation` = `stream` or `file` |

The container sends no data outside your account.

---

## Configuration

Everything has a default. You can override these through the `Environment` of the container in
`CreateModel`:

| Variable | Default | |
|----------|---------|---|
| `MAX_OPEN_STREAMS` | `32` | Concurrent streams per instance; further streams are refused. |
| `SM_MAX_CONCURRENT_TRANSFORMS` | `4` | Batch transform concurrency the container advertises. |
| `SM_BATCH_MAX_PAYLOAD_MB` | `25` | Batch record size the container advertises (concurrency x size must be at most 100). |
| `SM_MAX_PAYLOAD_MB` | `100` | Largest whole-file request. |
| `LOG_LEVEL` | `INFO` | |

---

## Samples

| Input | Duration | Streaming session | `InvokeEndpoint` | Batch transform |
|-------|----------|-------------------|------------------|-----------------|
| [`sample-10s.wav`](data/input/sample-10s.wav) | 10 s | [`sample-10s.jsonl`](data/output/streaming/sample-10s.jsonl) | [`sample-10s.json`](data/output/invocations/sample-10s.json) | [`sample-10s.wav.out`](data/output/batch/sample-10s.wav.out) |
| [`sample-30s.wav`](data/input/sample-30s.wav) | 30 s | [`sample-30s.jsonl`](data/output/streaming/sample-30s.jsonl) | [`sample-30s.json`](data/output/invocations/sample-30s.json) | [`sample-30s.wav.out`](data/output/batch/sample-30s.wav.out) |
| [`sample-60s.wav`](data/input/sample-60s.wav) | 60 s | [`sample-60s.jsonl`](data/output/streaming/sample-60s.jsonl) | [`sample-60s.json`](data/output/invocations/sample-60s.json) | - |

[`data/input/batch/`](data/input/batch) holds the batch transform inputs, and
[`data/input/batch-json/`](data/input/batch-json) a JSON envelope with options
([`sample-10s-keyterms.json.out`](data/output/batch-json/sample-10s-keyterms.json.out)).

Streaming sessions were captured with [`examples/stream_file.py`](examples/stream_file.py)
`--save`, whole-file results with [`examples/invoke_file.py`](examples/invoke_file.py) `--save`, and
batch results with [`examples/invoke_batch.py`](examples/invoke_batch.py), all with default options,
from the model package running on `ml.g6e.4xlarge` and (batch) `ml.g5.2xlarge`. Session ids and
timings differ from run to run.

## Sample audio attribution

The sample recordings are excerpts from the LibriVox recording of *Pride and Prejudice* by Jane
Austen (solo version), converted to 16 kHz mono 16-bit PCM WAV. LibriVox recordings are in the
public domain: https://archive.org/details/solo_pride_librivox

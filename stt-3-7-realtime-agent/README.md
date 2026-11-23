# STT 3.7 Realtime Agent - Input/Output Specification and Samples

Reference material for the **STT 3.7 Realtime Agent** speech-to-text model package on AWS
Marketplace. One endpoint serves:

- **live streaming** over `InvokeEndpointWithBidirectionalStream`: send audio as it is captured,
  receive partial and final transcripts as the speaker talks;
- **whole-file transcription** over `InvokeEndpoint`, asynchronous inference and batch transform.

STT 3.7 Realtime Agent is version 3.7.0 of AssemblyAI's realtime listing. It is built for voice agents: the model
detects the end of each turn itself, and if your agent runs its own turn detection, `mode=manual` lets it end each turn
and get the final transcript as soon as it sends `ForceEndpoint` (see
[Ending turns from the client](#ending-turns-from-the-client)). For versions 1.0.x
(Universal-3.6 Pro), see [`universal-3-6-pro-streaming/`](../universal-3-6-pro-streaming).

This folder contains:

- [`stt-3-7-realtime-agent-usage.ipynb`](stt-3-7-realtime-agent-usage.ipynb) - end-to-end
  sample notebook (subscribe, deploy, stream, transcribe a file, batch, usage metrics, clean up)
- [`examples/`](examples) - runnable Python examples
- [`data/input/`](data/input) - sample audio
- [`data/output/`](data/output) - the exact output each sample produces, captured from the model package
- [`requirements.txt`](requirements.txt) - Python packages for the examples

---

## Quick start

Before you start:

- Subscribe to the product on AWS Marketplace and copy the model package ARN for your region from
  the Configuration step. The ARN ends in a hash and is specific to the region.
- Have a SageMaker execution role (`AmazonSageMakerFullAccess`, plus read/write access to the S3
  bucket you use for batch transform).
- Set your region: `--region` on each example, or `AWS_DEFAULT_REGION` in the environment, to the
  region of the ARN. Check the service quota for the instance type you deploy (see Deployment).

```bash
pip install -r requirements.txt          # Python 3.12+

python examples/deploy.py --model-package-arn <arn> --role-arn <sagemaker-execution-role-arn>
python examples/stream_file.py stt-3-7-realtime-agent data/input/sample-10s.wav
python examples/stream_file.py stt-3-7-realtime-agent data/input/sample-30s.wav --force-endpoint   # voice-agent turns
python examples/invoke_file.py stt-3-7-realtime-agent data/input/sample-10s.wav
uv run examples/stream_mic.py stt-3-7-realtime-agent   # live from your microphone (needs uv: https://docs.astral.sh/uv/)

python examples/deploy.py --delete
```

| Example | What it does |
|---------|--------------|
| [`deploy.py`](examples/deploy.py) | Creates the model, endpoint configuration and endpoint; `--delete` removes all three |
| [`stream_file.py`](examples/stream_file.py) | Streams a WAV file in real time and prints partial and final turns; `--force-endpoint` ends each turn from the client |
| [`stream_mic.py`](examples/stream_mic.py) | Live-transcribes your microphone (a self-contained `uv run` script) |
| [`invoke_file.py`](examples/invoke_file.py) | Transcribes a whole file with `InvokeEndpoint` |
| [`invoke_batch.py`](examples/invoke_batch.py) | Transcribes files in S3 with a batch transform job |

---

## Deployment

| | |
|---|---|
| Real-time instance types | One GPU: `ml.g7e.2xlarge`, `ml.g7e.4xlarge`, `ml.g7e.8xlarge` (NVIDIA RTX PRO 6000); `ml.g6e.xlarge`, `ml.g6e.2xlarge`, `ml.g6e.4xlarge`, `ml.g6e.8xlarge`, `ml.g6e.16xlarge` (NVIDIA L40S); `ml.g7.2xlarge`, `ml.g7.4xlarge`, `ml.g7.8xlarge` (NVIDIA RTX PRO 4500). Several GPUs: `ml.g7e.12xlarge` (2x RTX PRO 6000), `ml.g7e.24xlarge` (4x), `ml.g7e.48xlarge` (8x); `ml.g6e.12xlarge`, `ml.g6e.24xlarge` (4x L40S), `ml.g6e.48xlarge` (8x); `ml.g7.12xlarge` (2x RTX PRO 4500), `ml.g7.24xlarge` (4x), `ml.g7.48xlarge` (8x). `ml.g7e.2xlarge` is the recommended starting point. Not `ml.g5` or `ml.p5` |
| Batch transform instance type | `ml.g6e.xlarge` |
| `InferenceAmiVersion` (real-time) | `al2023-ami-sagemaker-inference-gpu-4-1` - recommended |
| `TransformAmiVersion` (batch) | `al2-ami-sagemaker-batch-gpu-535`; pass no `Environment` |
| `ContainerStartupHealthCheckTimeoutInSeconds` | `3600` |
| `ModelDataDownloadTimeoutInSeconds` | `1800` |
| `RoutingConfig` (real-time) | `{"RoutingStrategy": "LEAST_OUTSTANDING_REQUESTS"}` - recommended |

The model needs CUDA 13 (NVIDIA driver 580 or later, or 535+ with CUDA compatibility libraries,
which the container enables itself). The `al2023-ami-sagemaker-inference-gpu-4-1` AMI provides
driver 580, so the container runs natively; on an older default driver it falls back to the
compatibility libraries. Check that your account has a quota for the instance type (for example
`ml.g7e.2xlarge for endpoint usage`); new accounts often start at 0.

The endpoint is `InService` once the weights are on the GPU and the model is compiled and warmed up; until then
`/ping` answers 503. The package carries the compiled model for RTX PRO 4500, so `ml.g7` starts in about 9 minutes.
On other GPUs the model compiles at startup, which took 14 to 26 minutes in our tests; hence the 3,600-second
startup timeout. Marketplace model packages run with network isolation and cannot be deployed to
Serverless Inference.

To let SageMaker pick from several sizes, give the production variant an `InstancePools` list instead of
a single `InstanceType`; `deploy.py --instance-type ml.g7e.2xlarge,ml.g7e.4xlarge` does this.

Set `RoutingConfig` to `LEAST_OUTSTANDING_REQUESTS` on the production variant, as `deploy.py` does. SageMaker
counts an open stream as outstanding for as long as it is open, so each new stream goes to the instance with
the fewest open streams, which keeps the load per instance even. Without it, SageMaker can send a new stream to a
busy instance while another is idle.

The routing balances by stream count, not by capacity, so keep every type in `InstancePools` in one GPU family and at
the same number of GPUs (for example `ml.g7e.2xlarge` and `ml.g7e.4xlarge`, one RTX PRO 6000 each). If one endpoint
mixes a 1-GPU and a 2-GPU instance, the 1-GPU instance carries twice the load per GPU.

On an instance with several GPUs, the container runs one copy of the model on each GPU and sends each
new stream or request to the GPU with the fewest open sessions; nothing else changes. The endpoint is
`InService` once every GPU is ready, so a multi-GPU instance takes longer to start. To use only some of the
GPUs, set `CUDA_VISIBLE_DEVICES` (see [Configuration](#configuration)).

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

`ModelQueryString` is limited to 2,048 characters once URL-encoded (a SageMaker limit), and encoding
roughly doubles a JSON list: quotes, brackets, commas and spaces take three characters each. A long
`keyterms_prompt` list or `prompt` may not fit, and SageMaker refuses the request before it reaches the
endpoint. Put the short settings (`sample_rate`, `encoding`, `language_codes`, `mode`) in the query string
and send the long ones in an `UpdateConfiguration` message as the first message on the stream, before any
audio. It has no such limit and applies to the whole session:

```json
{"type": "UpdateConfiguration", "keyterms_prompt": ["LibriVox", "SageMaker"], "prompt": "..."}
```

### Sending

| Payload | `DataType` | Contents |
|---------|-----------|----------|
| Audio | `BINARY` | Audio in the session's `encoding`. PCM: 20-1000 ms per message (50 ms is typical). Send at real time, as it is captured. |
| `{"type": "Terminate"}` | `UTF8` | Flush the last turn and end the session; the server replies with `Termination`. |
| `{"type": "ForceEndpoint"}` | `UTF8` | End the current turn now. |
| `{"type": "KeepAlive"}` | `UTF8` | Resets the `inactivity_timeout` timer; only needed when that parameter is set and no audio is being sent. Without it a session stays open until `Terminate` or the 30-minute stream limit, and the container answers SageMaker's transport pings itself. |
| `{"type": "UpdateConfiguration", ...}` | `UTF8` | Change `min_turn_silence`, `max_turn_silence`, `interruption_delay`, `vad_threshold`, `keyterms_prompt`, `prompt`, `agent_context` or `session_heartbeat` mid-session. |

### Receiving

Each event's payload is one JSON message ([`data/output/streaming/sample-10s.jsonl`](data/output/streaming/sample-10s.jsonl)
is a complete session):

| `type` | When |
|--------|------|
| `Begin` | The session is open; carries the session `id` and its effective `configuration`. |
| `SpeechStarted` | Speech was detected; carries its `timestamp` (ms into the audio). |
| `Turn` | A transcript update. `end_of_turn: false` is a partial that will change; `end_of_turn: true` is the final, formatted text of that turn. `turn_order` numbers the turns; `words` carries per-word `start`/`end` (ms) and `confidence`. |
| `Heartbeat` | Every 5 seconds of wall-clock time while `session_heartbeat` is on, during speech and silence: `total_audio_received_ms`, `total_duration_ms`, `realtime_factor` (audio received over wall-clock time since the previous heartbeat; about 1.0 when the client streams at real time) and `max_speech_probability` over that window. A liveness signal that also shows whether audio is being delivered at pace. |
| `Termination` | The session ended; carries `audio_duration_seconds`. |
| `Error` | The session was refused or failed; carries `error_code` and `error`. |

When the container closes a stream, SageMaker reports the close as a `ModelStreamError`, which the
Python SDK raises from `receive()`, **including after a normal `Termination`**. Treat the stream as
successful if `Termination` arrived; after an `Error`, the `ModelStreamError` is how the session ends.

Use one SDK client (one HTTP/2 connection) per concurrent stream; streams that share a client can stall.
Retry a stream that fails to open with HTTP 424: under heavy load SageMaker occasionally fails to
connect the stream or its metering channel to the container.

### Connection parameters

| Parameter | Description |
|-----------|-------------|
| `sample_rate` | Audio sample rate in Hz. Default `16000`. |
| `encoding` | `pcm_s16le` (16-bit little-endian mono PCM, default), `pcm_mulaw`, `opus` (raw Opus, one packet per message), `ogg_opus` (an Ogg Opus stream, in any chunks) or `aac` (an ADTS stream, in any chunks). `sample_rate` applies to the PCM encodings. |
| `speech_model` | Optional; if set, must be `assemblyai-stt-3-7-realtime-agent`. |
| `language_codes` | JSON list of up to 10 expected languages, e.g. `["en","es"]`. |
| `language_detection` | `true` to add `language_code` and `language_confidence` to each final turn. |
| `keyterms_prompt` | JSON list of up to 100 words and phrases to recognise, e.g. `["AssemblyAI","SageMaker"]`. |
| `prompt` | Free-text context about the audio (up to 1750 characters). |
| `agent_context` | What a voice agent just said, to improve recognition of the reply (up to 1750 characters). |
| `mode` | `balanced` (default), `min_latency`, `max_accuracy` or `manual`. With `manual`, a turn ends only when the client sends `ForceEndpoint` (see [Ending turns from the client](#ending-turns-from-the-client)). |
| `min_turn_silence` | Silence (ms) before the model runs an end-of-turn check; the turn ends if it reads as complete, otherwise it stays open. Raise it when an entity such as a number or an address is expected. |
| `max_turn_silence` | Silence (ms) after which a turn always ends, regardless of content; also the limit on how long an entity in progress is held open. |
| `interruption_delay` | How long (ms, 0-1000) after speech begins the first partial is emitted; lower for aggressive barge-in, higher for more confident first partials. Default set by `mode` (`min_latency` 0, `balanced` 500, `max_accuracy` 500). |
| `vad_threshold` | Confidence (0-1) needed to treat an audio frame as speech; default 0.2. Raise it in noisy environments or when the agent's own audio triggers interruptions, lower it for quiet speech. |
| `include_partial_turns` | `false` to receive final turns only. |
| `continuous_partials` | `true` to keep sending partials through long turns; `false` for one early partial per turn. |
| `inactivity_timeout` | Seconds of inactivity (5-3600) before the session ends; off by default. |
| `session_heartbeat` | `true` to receive a `Heartbeat` message every 5 seconds; off by default. Can also be switched on or off mid-session with `UpdateConfiguration`. |

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
| any other `speech_model` | the endpoint serves only `assemblyai-stt-3-7-realtime-agent` |

### Limits

| Limit | Value |
|-------|-------|
| Stream duration | 30 minutes per stream (a SageMaker limit); reconnect to continue |
| `ModelQueryString` | 2,048 characters, URL-encoded (a SageMaker limit); send long `keyterms_prompt`, `prompt` or `agent_context` in `UpdateConfiguration` |
| Audio per message | 20-1000 ms |
| Concurrent streams per GPU | Refused above 32 (RTX PRO 4500), 175 (L40S) or 350 (RTX PRO 6000); fixed in this version. An instance with N GPUs takes N times as many |

Capacity is per GPU. Recommended concurrent streams per GPU for latency-sensitive voice agents, from our sizing
tests (the final transcript about 150-250 ms after the end of speech, with `ForceEndpoint`):

| GPU | Streams per GPU |
|-----|-----------------|
| RTX PRO 6000 (`ml.g7e`) | 90 (an `ml.g7e.24xlarge` takes 360) |
| L40S (`ml.g6e`) | 40 |
| RTX PRO 4500 (`ml.g7`) | 25 |

In this version the endpoint refuses streams only at the fixed limits above, and `MAX_OPEN_STREAMS` has no effect.
On L40S and RTX PRO 6000 those limits are far above the recommendation, so an endpoint with too many streams keeps
accepting them and gets slower. Plan the number of instances for your peak from the recommendation, and scale up
with a multi-GPU instance or out with more instances and SageMaker auto scaling. With
`LEAST_OUTSTANDING_REQUESTS` routing (see [Deployment](#deployment)), new streams go to the least busy instance.

A stream over the limit opens normally and receives `Begin`; a few seconds later it receives an `Error` with
`error_code` 3005 ("Session Cancelled: An error occurred") and the stream ends, which the SDK reports as a
`ModelStreamError`. The refused stream is not billed. Retry it with a short backoff; the retry is routed again.

### Ending turns from the client

By default the model detects the end of each turn itself; `mode` (`balanced`, `min_latency` or `max_accuracy`) sets
how it trades latency against accuracy. If your voice agent runs its own turn detection, such as a voice activity
detector or push-to-talk, open the stream with `mode=manual` instead and send `{"type": "ForceEndpoint"}` when your
detector fires: the model ends the turn
there and the final transcript (`end_of_turn: true`) comes back right away, without waiting for silence of its
own. Each `ForceEndpoint` ends exactly one turn.

```
ModelQueryString: sample_rate=16000&encoding=pcm_s16le&mode=manual
on end of speech: {"type": "ForceEndpoint"}
```

`examples/stream_file.py --force-endpoint` does this with a simple energy detector: it sends `ForceEndpoint`
after `--silence-ms` (default 600) of quiet that follows speech. A detector that fires on short pauses splits one
utterance into several turns; let your agent's own end-of-turn logic decide when to answer. A `ForceEndpoint` sent
after a noise or a breath, with no words in the turn, returns a final turn with an empty `transcript`; skip it. Send the agent's last
reply as `agent_context` in `UpdateConfiguration` so the model knows what the user is answering.

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
    EndpointName="stt-3-7-realtime-agent",
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
- Run the job on `ml.g6e.xlarge`; it needs the `ml.g6e.xlarge for transform job usage` quota in your account,
  which starts at 0 in most accounts.
- Leave `MaxConcurrentTransforms` and `MaxPayloadInMB` unset to use the container's defaults: 4
  concurrent records of up to 25 MB each.
- A record must finish within `ModelClientConfig.InvocationsTimeoutInSeconds` (default 600, maximum
  3600); raise it for long recordings.
- A job takes about 10 minutes before its first record, whatever the input size: the instance starts and
  the weights load, as for an endpoint. Batch suits backlogs, not a handful of files.
- Batch transform cannot set `CustomAttributes`. To pass options, store each file as a JSON envelope
  (`{"audio": "<base64>", "config": {...}}`) and run the job with `ContentType="application/json"`;
  [`data/input/batch-json/`](data/input/batch-json) has an example.

---

## Billing and usage metrics

The software charge is metered by the amount of audio processed, in units of **one hour of audio**
(see the listing for the price per unit, which is the price per audio hour). Usage is counted to the
millisecond and billed as a decimal number of units, never rounded up.

- A stream is billed for the duration of the audio sent to it from when the session begins until
  `Terminate` (for Opus and AAC, the duration the frames encode), reported as the stream runs.
- A whole-file `InvokeEndpoint` request is billed for the file's duration, at least 0.000001 unit.
- A session that is refused, or that fails before its first transcript, is not billed. A request that
  fails is not billed. A stream that drops after transcribing is billed for the audio it received.

The AWS infrastructure charge for the instance is separate.

Batch transform is priced differently: per instance-hour (see the listing), with no per-audio charge.

Every stream and request also writes its usage to the container's CloudWatch logs (the endpoint's or
the transform job's) in Embedded Metric
Format, so you can graph, alarm on and reconcile usage in your own account:

| | |
|---|---|
| Namespace | `AssemblyAI/STT-3.7-Realtime-Agent` |
| Metrics | `ConsumedUnits` (billed units, in audio hours), `AudioDurationSeconds`, `UnmeteredAudioSeconds` (audio whose usage could not be reported to SageMaker, so was not billed; normally 0) |
| Dimension | `Operation` = `stream` or `file` |

The container sends no data outside your account.

---

## Configuration

Everything has a default. You can override these through the `Environment` of the container in
`CreateModel`:

| Variable | Default | |
|----------|---------|---|
| `CUDA_VISIBLE_DEVICES` | all GPUs | The GPUs to serve on, e.g. `0,1`; one copy of the model runs on each. |
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

[`sample-30s-force-endpoint.jsonl`](data/output/streaming/sample-30s-force-endpoint.jsonl) is the 30 s sample streamed
with `--force-endpoint`: each pause ends a turn, and two `ForceEndpoint` messages sent after a pause with no new words
return empty final turns.

[`data/input/batch/`](data/input/batch) holds the batch transform inputs, and
[`data/input/batch-json/`](data/input/batch-json) a JSON envelope with options
([`sample-10s-keyterms.json.out`](data/output/batch-json/sample-10s-keyterms.json.out)).

Streaming sessions were captured with [`examples/stream_file.py`](examples/stream_file.py)
`--save`, whole-file results with [`examples/invoke_file.py`](examples/invoke_file.py) `--save`, and
batch results with [`examples/invoke_batch.py`](examples/invoke_batch.py), all with default options,
from the model package running on `ml.g7.2xlarge` and (batch) `ml.g6e.xlarge`. Session ids and
timings differ from run to run.

## Sample audio attribution

The sample recordings are excerpts from the LibriVox recording of *Pride and Prejudice* by Jane
Austen (solo version), converted to 16 kHz mono 16-bit PCM WAV. LibriVox recordings are in the
public domain: https://archive.org/details/solo_pride_librivox

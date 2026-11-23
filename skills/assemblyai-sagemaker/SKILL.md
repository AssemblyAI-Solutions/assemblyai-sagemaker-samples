---
name: assemblyai-sagemaker
description: Use when deploying, calling, scaling or troubleshooting AssemblyAI-STT-3.7-Realtime-Agent or Universal-3.6 Pro speech-to-text on Amazon SageMaker AI from their AWS Marketplace model packages. Covers creating the model, endpoint configuration and endpoint; streaming audio with InvokeEndpointWithBidirectionalStream; transcribing files with InvokeEndpoint or batch transform; auto scaling; and debugging endpoints that fail to start, hanging streams, 424s, ModelStreamError, and error codes 3005, 3006 or 3007.
metadata:
  author: assemblyai
  version: "1.1"
---

# AssemblyAI streaming speech-to-text on Amazon SageMaker

AssemblyAI-STT-3.7-Realtime-Agent and Universal-3.6 Pro each run as an AWS Marketplace model package in the customer's own AWS account. An endpoint serves the model in the package it was deployed from; ask the user which model they subscribed to. A stream is one SageMaker `InvokeEndpointWithBidirectionalStream` request that carries AssemblyAI's v3 streaming protocol (`Begin`, `Turn`, `Termination`, `Error`). Authentication is AWS IAM (SigV4), not an AssemblyAI API key.

The full guide is at `https://www.assemblyai.com/docs/self-hosted/sagemaker/streaming`; fetch it as Markdown from `https://www.assemblyai.com/docs/self-hosted/sagemaker/streaming.md`. Copy code from that page instead of writing SageMaker streaming code from memory: the streaming SDK is new, experimental and easy to get wrong.

## Before creating anything

1. **Resolve the AWS context.** Run `aws sts get-caller-identity` and read the Region from the user's configuration. Don't assume `us-east-1`.
2. **Get the model package ARN from the user.** There's no API to subscribe or to look the ARN up. The user subscribes to the listing in the AWS Marketplace console, chooses **Continue to configuration**, picks the version and Region, and copies the ARN (later: **Manage subscriptions**). The ARN is Region-specific (`arn:aws:sagemaker:<region>:<aws-owned-account>:model-package/<package-name>`); its Region must match the endpoint's.
3. **Check quota** for each instance type you'll use. New accounts often have 0:
   ```bash
   aws service-quotas list-service-quotas --service-code sagemaker --region "$AWS_REGION" \
     --query "Quotas[?QuotaName=='ml.g7e.2xlarge for endpoint usage'].Value"
   ```
4. **Find or create an execution role** trusted by `sagemaker.amazonaws.com` with `AmazonSageMakerFullAccess`. Look for an existing one first: `aws iam list-roles --query "Roles[?contains(RoleName, 'SageMaker')].Arn"` (often under the `service-role/` path). SSO users often can't create or inspect IAM roles; then ask the user for the role ARN.
5. **Confirm cost with the user** before `CreateEndpoint`: GPU instance-hours from AWS, plus AssemblyAI's usage fee per hour of audio (see the listing for the price).

## Deploy

Create a model, an endpoint configuration and an endpoint, with these settings. Use the AWS CLI, an SDK or infrastructure as code: the SageMaker console can't set `InferenceAmiVersion`.

```python
import boto3

sm = boto3.client("sagemaker", region_name=REGION)
sm.create_model(
    ModelName=NAME,
    PrimaryContainer={"ModelPackageName": MODEL_PACKAGE_ARN},  # optional: "Environment": {"MAX_OPEN_STREAMS": "8"}
    ExecutionRoleArn=ROLE_ARN,
    EnableNetworkIsolation=True,  # required for Marketplace model packages
)
sm.create_endpoint_config(
    EndpointConfigName=NAME,
    ProductionVariants=[{
        "VariantName": "primary",
        "ModelName": NAME,
        "InitialInstanceCount": 1,
        "InstancePools": [  # tried in order; every type must have the same number of GPUs
            {"InstanceType": "ml.g7e.2xlarge", "Priority": 1},
            {"InstanceType": "ml.g7e.4xlarge", "Priority": 2},
            {"InstanceType": "ml.g6e.4xlarge", "Priority": 3},
        ],
        "InferenceAmiVersion": "al2023-ami-sagemaker-inference-gpu-4-1",  # required: NVIDIA driver for CUDA 13
        "RoutingConfig": {"RoutingStrategy": "LEAST_OUTSTANDING_REQUESTS"},
        "ModelDataDownloadTimeoutInSeconds": 1800,
        "ContainerStartupHealthCheckTimeoutInSeconds": 1800,
    }],
)
sm.create_endpoint(EndpointName=NAME, EndpointConfigName=NAME)
sm.get_waiter("endpoint_in_service").wait(EndpointName=NAME, WaiterConfig={"Delay": 30, "MaxAttempts": 120})
```

A healthy endpoint is usually `InService` in 5 to 20 minutes; AssemblyAI-STT-3.7-Realtime-Agent compiles the model at startup on every GPU but `ml.g7` and can take up to 30, so give it a 3,600-second `ContainerStartupHealthCheckTimeoutInSeconds`. One that can't get GPU capacity can take 30 minutes or more to end in `Failed`. On `Failed`, read `describe_endpoint(...)["FailureReason"]` and the log group `/aws/sagemaker/Endpoints/<name>`. A `Failed` within a few minutes with `Request to service failed...` and no log group is transient: delete the endpoint and create it again from the same configuration. With a pool, `describe_endpoint(...)["ProductionVariants"][0]["InstancePools"]` shows the type it got.

**Instance types.** `ml.g7e.2xlarge` is the recommended start. The listing shows which types each version supports; AssemblyAI-STT-3.7-Realtime-Agent doesn't run on `ml.g5` or `ml.p5`.

| GPUs | Types |
|---|---|
| 1 | `ml.g7e.2xlarge`, `4xlarge`, `8xlarge`; `ml.g7.2xlarge`, `4xlarge`, `8xlarge`; `ml.g6e.xlarge`, `2xlarge`, `4xlarge`, `8xlarge`, `16xlarge`; `ml.g5.xlarge`, `2xlarge`, `4xlarge` (Universal-3.6 Pro only); `ml.p5.4xlarge` (Universal-3.6 Pro only) |
| 2 | `ml.g7e.12xlarge`; `ml.g7.12xlarge` |
| 4 | `ml.g7e.24xlarge`; `ml.g7.24xlarge`; `ml.g6e.12xlarge`, `24xlarge`; `ml.g5.12xlarge`, `24xlarge` (Universal-3.6 Pro only) |
| 8 | `ml.g7e.48xlarge`; `ml.g7.48xlarge`; `ml.g6e.48xlarge`; `ml.g5.48xlarge` (Universal-3.6 Pro only); `ml.p5.48xlarge` (Universal-3.6 Pro only) |

**Capacity.** Streams and whole-file requests both count, per GPU, and a multi-GPU instance serves on every GPU.
- AssemblyAI-STT-3.7-Realtime-Agent: recommended per GPU for voice agents `ml.g7e` 90, `ml.g6e` 40, `ml.g7` 25. The endpoint refuses sessions only above a fixed limit of 350, 175 and 32 per GPU, and `MAX_OPEN_STREAMS` has no effect, so size the instance count for the recommendation; past it the endpoint keeps accepting and gets slower.
- Universal-3.6 Pro: each GPU serves `MAX_OPEN_STREAMS` sessions (default 32). Recommended `ml.g7e` 64 to 96; `ml.g7` 16 to 24; `ml.g6e` 32 (16 on `xlarge`, 24 on `2xlarge` and `12xlarge`); `ml.g5` 8, which must be set because the default 32 is too many. Change it with the model's `Environment`; on a running endpoint, create a new model and endpoint configuration and call `UpdateEndpoint`.

## Stream audio

Use the `stream.py` in the guide's **Stream audio** section. The rules it follows:

- Python 3.12+ (for example `uv venv --python 3.12`), with these exact versions (newer releases stall the input stream):
  `pip install "aws-sdk-sagemaker-runtime-http2==0.11.0" "smithy-http==0.5.0" "awscrt==0.32.2" "boto3>=1.40"`
- Endpoint URI `https://runtime.sagemaker.<region>.amazonaws.com:8443`. Without `:8443` the client hangs. `stream.py` takes credentials and the Region from the AWS configuration, so set `AWS_PROFILE` and `AWS_DEFAULT_REGION` when running it.
- Session options go in `model_query_string`, built with `urllib.parse.urlencode(options, quote_via=urllib.parse.quote)`. A `+` for a space is rejected. The limit is 2,048 characters; send a long `prompt` or `keyterms_prompt` in an `{"type": "UpdateConfiguration", ...}` message before any audio.
- Audio is `BINARY` payload parts; control messages (`Terminate`, `ForceEndpoint`, `UpdateConfiguration`, `KeepAlive`) are `UTF8` JSON. Send PCM in 20 to 1,000 ms messages (50 ms is typical) at real-time pace.
- One SDK client per concurrent stream. Streams that share a client stall.
- End with `Terminate` and read until `Termination`. The SDK then raises `ModelStreamError` from `receive()`: that's the normal close, not a failure. It also logs `AWS_ERROR_HTTP_STREAM_HAS_COMPLETED` from a background task at shutdown unless you install the exception handler `stream.py` uses.
- A stream lasts at most 30 minutes. Open a new one before then.
- `language_codes` takes specific codes, such as `["en","es"]`; `multi` isn't supported.

| Signal | Meaning | Action |
|---|---|---|
| `Begin`, then `Error` 3005 a few seconds later | Instance at its stream limit, or a server error. Not billed. | Retry with backoff; add capacity if frequent. |
| `Error` 3006 | Invalid or unavailable option | Fix the option named in the message. |
| `Error` 3007 | Empty or over-1,000 ms PCM message, or audio sent faster than real time | Fix framing and pacing. |
| `ModelError` 424 when opening | SageMaker couldn't connect to the container | Retry. |
| Hang with no response | Missing `:8443` | Fix the endpoint URI. |

## Transcribe files

`InvokeEndpoint` on the same endpoint returns every turn at once (`text`, `turns`, `words`). `ContentType` is any `audio/*` (WAV, FLAC, OGG, MP3), `audio/l16` (raw PCM, set `sample_rate`), or `application/json` (`{"audio": "<base64>", "config": {...}}`). Pass options as JSON in `CustomAttributes` (1,024 characters at most). SageMaker allows 60 seconds per request, so use batch transform for long files: `ml.g6e.xlarge` for AssemblyAI-STT-3.7-Realtime-Agent or `ml.g5.2xlarge` for Universal-3.6 Pro, `SplitType` `None`, `TransformAmiVersion` `al2-ami-sagemaker-batch-gpu-535`, and no `Environment`. Batch transform details: the README in the samples repository's folder for the model, `https://github.com/AssemblyAI-Solutions/assemblyai-sagemaker-samples` (`stt-3-7-realtime-agent/` or `universal-3-6-pro-streaming/`).

## Scale and monitor

- Route with `LEAST_OUTSTANDING_REQUESTS`, which counts open streams. With random routing, streams get refused while other instances have room.
- Auto scaling: Application Auto Scaling target tracking on `SageMakerVariantConcurrentRequestsPerModelHighResolution`, with a target below the recommended streams per instance (the per-GPU figure × GPUs). Not `GPUUtilization` or `InvocationsPerInstance`. Real-time endpoints don't scale to zero.
- Billing is per hour of audio transcribed, not per instance-hour, plus AWS's instance charge. Usage is in CloudWatch in the user's account: metric `ConsumedUnits`, namespace `AssemblyAI/STT-3.7-Realtime-Agent` or `AssemblyAI/Universal-3.6-Pro-Realtime`.

## Tear down

Deregister any scalable target, then delete the endpoint, the endpoint configuration and the model, and confirm with `aws sagemaker list-endpoints`. Billing stops when the endpoint is deleted. The Marketplace subscription stays and costs nothing while unused.

## Common mistakes

| Mistake | Result |
|---|---|
| No `InferenceAmiVersion` | Container fails at startup with a CUDA or driver error on some families, such as `ml.g5` |
| Creating the endpoint in the console | `InferenceAmiVersion` can't be set |
| Default startup timeouts | Health check times out while the model loads (AssemblyAI-STT-3.7-Realtime-Agent needs 3,600 seconds) |
| AssemblyAI-STT-3.7-Realtime-Agent on `ml.g5` or `ml.p5` | Container fails at startup: it runs out of GPU memory |
| One `InstanceType`, no pool | `InsufficientInstanceCapacity` when that type runs short |
| Pool mixing GPU counts | Smaller instances fill first and refuse streams |
| Latest or guessed streaming SDK | Input stalls, `Termination` never arrives, or import errors |
| Treating `ModelStreamError` after `Termination` as a failure | False errors and retries |
| `MAX_OPEN_STREAMS` 32 on `ml.g5` (Universal-3.6 Pro) | Out-of-memory errors and refused streams under load |
| Relying on `MAX_OPEN_STREAMS` with AssemblyAI-STT-3.7-Realtime-Agent | No effect; too few instances shows up as latency, not refusals |

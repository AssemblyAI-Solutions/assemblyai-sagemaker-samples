#!/usr/bin/env python3
"""Transcribe audio with a Universal 3.5 Pro Sync real-time endpoint.

Two request forms are shown:
  * raw audio bytes  — simplest, uses defaults
  * multipart        — required to pass configuration options such as timestamps

Usage:
    python invoke_realtime.py --endpoint <name> --audio sample.wav [--timestamps]
    python invoke_realtime.py --endpoint <name> --audio sample.wav \
        --config '{"timestamps": true, "language_code": "en", "keyterms": ["LibriVox"]}'
"""
import argparse
import json
import uuid

import boto3


def build_multipart(audio: bytes, config: dict, filename: str = "audio.wav"):
    """Build a multipart/form-data body with `audio` and `config` parts.

    Returns (body_bytes, content_type).
    """
    boundary = f"----boundary{uuid.uuid4().hex}"
    body = b"".join([
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="audio"; filename="{filename}"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode(),
        audio,
        (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="config"\r\n'
            f"Content-Type: application/json\r\n\r\n"
            f"{json.dumps(config)}\r\n"
            f"--{boundary}--\r\n"
        ).encode(),
    ])
    return body, f"multipart/form-data; boundary={boundary}"


def transcribe(client, endpoint: str, audio: bytes, config: dict | None = None) -> dict:
    if config is None:
        body, content_type = audio, "audio/wav"
    else:
        body, content_type = build_multipart(audio, config)

    response = client.invoke_endpoint(
        EndpointName=endpoint, ContentType=content_type, Body=body
    )
    return json.loads(response["Body"].read())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--audio", required=True, help="path to a WAV file (<= 120 seconds)")
    p.add_argument("--timestamps", action="store_true", help="request word-level timings")
    p.add_argument(
        "--config",
        default=None,
        help="JSON object of configuration options (see the README table); "
        "merged with --timestamps",
    )
    p.add_argument("--region", default=None)
    args = p.parse_args()

    client = boto3.client("sagemaker-runtime", region_name=args.region)
    with open(args.audio, "rb") as f:
        audio = f.read()

    config = json.loads(args.config) if args.config else {}
    if args.timestamps:
        config["timestamps"] = True
    config = config or None
    result = transcribe(client, args.endpoint, audio, config)

    print(result["text"])
    print()
    print(
        f"confidence {result['confidence']:.3f} | "
        f"audio {result['audio_duration_ms']} ms | "
        f"server {result['request_time_ms']:.0f} ms"
    )

    if args.timestamps:
        print()
        print(f"{'start':>8} {'end':>8}  word")
        for w in result["words"]:
            # start/end are omitted for any word that could not be aligned.
            if "start" in w:
                print(f"{w['start']:>8} {w['end']:>8}  {w['text']}")
            else:
                print(f"{'-':>8} {'-':>8}  {w['text']}")


if __name__ == "__main__":
    main()

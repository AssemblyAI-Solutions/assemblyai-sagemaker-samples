#!/usr/bin/env python3
"""Transcribe a whole audio file with InvokeEndpoint (POST /invocations).

Sends the file as the request body; session options go in the
X-Amzn-SageMaker-Custom-Attributes header as a JSON object.

Usage:
    python invoke_file.py <endpoint-name> <audio-file> [--config '{"keyterms_prompt": ["LibriVox"]}']

Accepts WAV, FLAC, OGG and MP3 (8-48 kHz, any channel count). InvokeEndpoint
allows 60 seconds per request, which suits files of about a minute; use
asynchronous inference or batch transform for longer audio. CustomAttributes
is limited to 1024 characters.
"""
import argparse
import json
import mimetypes

import boto3
from botocore.config import Config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("endpoint")
    p.add_argument("audio")
    p.add_argument("--config", default=None, help="session options as a JSON object (see the README)")
    p.add_argument("--save", help="write the JSON response to this file")
    p.add_argument("--profile", default=None)
    p.add_argument("--region", default=None)
    args = p.parse_args()

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    # Wait out InvokeEndpoint's 60 s limit, and don't retry: a retried request
    # is transcribed (and billed) again.
    smr = session.client("sagemaker-runtime", config=Config(read_timeout=70, retries={"max_attempts": 0}))

    with open(args.audio, "rb") as f:
        body = f.read()
    request = {
        "EndpointName": args.endpoint,
        "ContentType": mimetypes.guess_type(args.audio)[0] or "application/octet-stream",
        "Body": body,
    }
    if args.config:
        json.loads(args.config)  # fail early on malformed JSON
        request["CustomAttributes"] = args.config

    try:
        response = smr.invoke_endpoint(**request)
    except smr.exceptions.ModelError as e:
        # The container's 4xx answers (refused options, unreadable audio) arrive here.
        raise SystemExit(f"HTTP {e.response.get('OriginalStatusCode')}: {e.response.get('OriginalMessage')}")

    result = json.loads(response["Body"].read())
    print(result["text"])
    print(f"\n{len(result['turns'])} turns, {result['audio_duration_ms']} ms of audio, session {result['session_id']}")
    if args.save:
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(f"Saved the response to {args.save}")


if __name__ == "__main__":
    main()

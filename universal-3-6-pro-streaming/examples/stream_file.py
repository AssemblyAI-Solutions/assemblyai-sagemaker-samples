#!/usr/bin/env python3
"""Stream a WAV file to a Universal-3.6 Pro Streaming endpoint in real time.

Sends the audio in 50 ms frames over InvokeEndpointWithBidirectionalStream,
prints each partial and final turn as it arrives, and ends the session with a
Terminate message.

Usage:
    python stream_file.py <endpoint-name> <file.wav> [--param key=value ...] [--save messages.jsonl]

    python stream_file.py my-endpoint ../data/input/sample-10s.wav \\
        --param 'keyterms_prompt=["LibriVox"]' --param mode=min_latency

The WAV must be 16-bit PCM, mono, 8-48 kHz. Credentials come from the default
AWS chain (or --profile). Requires Python 3.12+ and requirements.txt.
"""
import argparse
import asyncio
import contextlib
import json
import time
import urllib.parse
import wave

import boto3
from aws_sdk_sagemaker_runtime_http2.client import AsyncSageMakerRuntimeHTTP2Client
from aws_sdk_sagemaker_runtime_http2.config import AsyncSageMakerRuntimeHTTP2Config
from aws_sdk_sagemaker_runtime_http2.models import (
    InvokeEndpointWithBidirectionalStreamInput,
    RequestPayloadPart,
    RequestStreamEventPayloadPart,
)
from smithy_http.aio.crt import AWSCRTHTTPClient

FRAME_MS = 50  # each audio message must hold 20-1000 ms of audio
DRAIN_TIMEOUT_S = 10  # how long to wait for Termination after the audio ends


def read_wav(path):
    with wave.open(path) as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise SystemExit(f"{path}: need 16-bit mono PCM (e.g. ffmpeg -i in -ac 1 -ar 16000 -sample_fmt s16 out.wav)")
        return w.readframes(w.getnframes()), w.getframerate()


def payload(data, data_type):
    return RequestStreamEventPayloadPart(value=RequestPayloadPart(bytes_=data, data_type=data_type))


def quiet_sdk_close(loop, context):
    # After the server ends a stream the SDK still tries to close the request
    # body in the background, which fails harmlessly.
    if "HTTP_STREAM_HAS_COMPLETED" not in str(context.get("exception", "")):
        loop.default_exception_handler(context)


async def stream_file(endpoint, pcm, sample_rate, params, region, credentials, on_message):
    """Stream 16-bit mono PCM in real time; call on_message for each server message.

    Returns True if the session ended with a Termination message.
    """
    config = await AsyncSageMakerRuntimeHTTP2Config.resolve(
        region=region,
        endpoint_uri=f"https://runtime.sagemaker.{region}.amazonaws.com:8443",
        transport=AWSCRTHTTPClient(),
        aws_access_key_id=credentials.access_key,
        aws_secret_access_key=credentials.secret_key,
        aws_session_token=credentials.token,
    )
    query = urllib.parse.urlencode(
        {"sample_rate": sample_rate, "encoding": "pcm_s16le", **params},
        # SageMaker rejects "+" for a space in ModelQueryString; use %20.
        quote_via=urllib.parse.quote,
    )
    asyncio.get_running_loop().set_exception_handler(quiet_sdk_close)

    # Use one client per stream: streams that share a client (one HTTP/2
    # connection) can stall.
    async with AsyncSageMakerRuntimeHTTP2Client(config=config) as client:
        stream = await client.invoke_endpoint_with_bidirectional_stream(
            InvokeEndpointWithBidirectionalStreamInput(endpoint_name=endpoint, model_query_string=query)
        )

        async def send():
            frame = sample_rate * 2 * FRAME_MS // 1000
            start = time.monotonic()
            for i, offset in enumerate(range(0, len(pcm), frame)):
                await stream.input_stream.send(payload(pcm[offset:offset + frame], "BINARY"))
                # Pace the audio at real time, as a live source would.
                await asyncio.sleep(max(0, start + (i + 1) * FRAME_MS / 1000 - time.monotonic()))
            # Flush the last turn and end the session.
            await stream.input_stream.send(payload(json.dumps({"type": "Terminate"}).encode(), "UTF8"))

        sender = asyncio.create_task(send())
        terminated = False
        try:
            _, output = await stream.await_output()
            while True:
                timeout = DRAIN_TIMEOUT_S if sender.done() else None
                try:
                    event = await asyncio.wait_for(output.receive(), timeout)
                except Exception as e:  # noqa: BLE001
                    # SageMaker reports the container closing the stream (even
                    # normally) as a ModelStreamError; after an Error message
                    # this is how a refused session ends.
                    print(f"Stream ended: {type(e).__name__}")
                    break
                value = getattr(event, "value", None)
                if not hasattr(value, "bytes_"):
                    print("Stream ended:", event)
                    break
                message = json.loads(value.bytes_)
                on_message(message)
                if message.get("type") == "Termination":
                    terminated = True
                    break
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(stream.close(), 2)
    return terminated


def print_message(message):
    kind = message.get("type")
    if kind == "Begin":
        print(f"Begin: session {message['id']}")
    elif kind == "Turn":
        label = "FINAL  " if message.get("end_of_turn") else "partial"
        print(f"[{message['turn_order']}] {label} {message['transcript']}")
    elif kind == "SpeechStarted":
        print(f"Speech started at {message['timestamp']} ms")
    elif kind == "Termination":
        print(f"Termination: {message.get('audio_duration_seconds')} s of audio")
    elif kind == "Error":
        print(f"Error {message.get('error_code')}: {message.get('error')}")
    else:
        print(message)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("endpoint")
    p.add_argument("wav")
    p.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                   help="connection parameter, repeatable (see the README)")
    p.add_argument("--save", help="write every server message to this JSON Lines file")
    p.add_argument("--profile", default=None)
    p.add_argument("--region", default=None)
    args = p.parse_args()

    if any("=" not in item for item in args.param):
        p.error("--param takes KEY=VALUE")
    params = dict(item.split("=", 1) for item in args.param)
    pcm, sample_rate = read_wav(args.wav)
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    if session.get_credentials() is None:
        p.error("no AWS credentials found; configure a profile or pass --profile")
    if not session.region_name:
        p.error("no AWS Region configured; pass --region (the endpoint's Region)")
    credentials = session.get_credentials().get_frozen_credentials()
    region = session.region_name

    saved = []

    def on_message(message):
        saved.append(message)
        print_message(message)

    ok = asyncio.run(stream_file(args.endpoint, pcm, sample_rate, params, region, credentials, on_message))
    if args.save:
        with open(args.save, "w") as f:
            f.writelines(json.dumps(m) + "\n" for m in saved)
        print(f"Saved {len(saved)} messages to {args.save}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "aws-sdk-sagemaker-runtime-http2==0.11.0",
#     "smithy-http==0.5.0",
#     "awscrt==0.32.2",
#     "boto3>=1.40",
#     "sounddevice>=0.5",
# ]
# ///
"""Live-transcribe your microphone through a Universal-3.6 Pro Streaming endpoint.

    uv run stream_mic.py <endpoint-name> [--profile P] [--region R] [--query 'k=v&...']

Speak; the current turn updates in place (grey) and each finished turn is
printed in full. Ctrl+C ends the session cleanly. SageMaker caps one
bidirectional stream at 30 minutes.

Credentials come from the AWS profile (`--profile`, else AWS_PROFILE, else the
default chain), so log in first: `aws sso login --profile <profile>`. On macOS
the terminal app needs microphone permission (System Settings > Privacy &
Security > Microphone). `--list-devices` shows the inputs; `--device N` picks one.
`--wav FILE` plays a 16 kHz mono 16-bit WAV in real time instead of the mic.
Runs on macOS and Linux.
"""

import argparse
import asyncio
import contextlib
import json
import shutil
import signal
import sys
import wave

import boto3
import sounddevice as sd
from aws_sdk_sagemaker_runtime_http2.client import AsyncSageMakerRuntimeHTTP2Client
from aws_sdk_sagemaker_runtime_http2.config import AsyncSageMakerRuntimeHTTP2Config
from aws_sdk_sagemaker_runtime_http2.models import (
    InvokeEndpointWithBidirectionalStreamInput,
    RequestPayloadPart,
    RequestStreamEventPayloadPart,
)
from smithy_http.aio.crt import AWSCRTHTTPClient

SAMPLE_RATE = 16000
FRAME_MS = 50  # each audio message must hold 20-1000 ms of audio

TTY = sys.stdout.isatty()
# Colours and the in-place partial line only make sense on a terminal.
GREY, BOLD, RED, RESET, CLEAR_LINE = (
    ("\x1b[90m", "\x1b[1m", "\x1b[31m", "\x1b[0m", "\r\x1b[2K") if TTY else ("",) * 5
)
# How long to wait for the server to finish after we stop sending.
DRAIN_TIMEOUT_S = 5
# How long to wait for SDK teardown, which can block once the server is gone.
CLOSE_TIMEOUT_S = 2


def _payload(data: bytes, data_type: str) -> RequestStreamEventPayloadPart:
    return RequestStreamEventPayloadPart(value=RequestPayloadPart(bytes_=data, data_type=data_type))


def _show_partial(text: str) -> None:
    if not TTY:
        return
    # Keep the in-place line within one terminal row, or \r can't overwrite it.
    width = shutil.get_terminal_size().columns - 4
    if len(text) > width:
        text = "…" + text[-(width - 1) :]
    sys.stdout.write(f"{CLEAR_LINE}{GREY}{text}{RESET}")
    sys.stdout.flush()


def _say(text: str) -> None:
    """Print on a fresh line, clearing any partial being shown."""
    sys.stdout.write(f"{CLEAR_LINE}{text}\n")
    sys.stdout.flush()


def _quiet_sdk_close(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    # After the server ends the stream, the SDK still tries to end the request
    # body in a background task, which fails harmlessly; don't print it.
    if "HTTP_STREAM_HAS_COMPLETED" in str(context.get("exception", "")):
        return
    loop.default_exception_handler(context)


async def _bounded(awaitable, timeout: float) -> None:
    with contextlib.suppress(Exception):
        await asyncio.wait_for(awaitable, timeout)


async def run(args: argparse.Namespace) -> int:
    try:
        session = boto3.Session(profile_name=args.profile, region_name=args.region)
        creds = session.get_credentials()
        creds = creds.get_frozen_credentials() if creds else None
    except Exception as e:  # noqa: BLE001 - expired SSO, unknown profile, ...
        _say(f"{RED}AWS credentials: {type(e).__name__}: {e}{RESET}")
        creds = None
    if creds is None:
        _say(f"{RED}No AWS credentials; run `aws sso login --profile <profile>` first.{RESET}")
        return 1
    region = session.region_name
    if not region:
        _say(f"{RED}No AWS Region configured; pass --region (the endpoint's Region).{RESET}")
        return 1

    config = await AsyncSageMakerRuntimeHTTP2Config.resolve(
        region=region,
        endpoint_uri=f"https://runtime.sagemaker.{region}.amazonaws.com:8443",
        transport=AWSCRTHTTPClient(),
        aws_access_key_id=creds.access_key,
        aws_secret_access_key=creds.secret_key,
        aws_session_token=creds.token,
    )
    query = f"sample_rate={SAMPLE_RATE}&encoding=pcm_s16le"
    if args.query:
        query += "&" + args.query.lstrip("&")

    loop = asyncio.get_running_loop()
    loop.set_exception_handler(_quiet_sdk_close)
    # Audio frames for the sender; None means "stop sending".
    audio: asyncio.Queue[bytes | None] = asyncio.Queue()
    stopping = asyncio.Event()

    def stop() -> None:
        if not stopping.is_set():
            stopping.set()
            audio.put_nowait(None)

    def on_sigint() -> None:
        stop()
        # A second Ctrl+C gets Python's default handling and exits at once.
        loop.remove_signal_handler(signal.SIGINT)

    loop.add_signal_handler(signal.SIGINT, on_sigint)

    def on_audio(indata, frames, time_info, status) -> None:  # runs on the audio thread
        loop.call_soon_threadsafe(audio.put_nowait, bytes(indata))

    _say(f"Connecting to {args.endpoint} ({region})…")
    async with AsyncSageMakerRuntimeHTTP2Client(config=config) as client:
        try:
            stream = await client.invoke_endpoint_with_bidirectional_stream(
                InvokeEndpointWithBidirectionalStreamInput(
                    endpoint_name=args.endpoint, model_query_string=query
                )
            )
        except Exception as e:  # noqa: BLE001 - wrong endpoint, permissions, ...
            _say(f"{RED}Could not open the stream: {type(e).__name__}: {e}{RESET}")
            return 1

        async def play_wav() -> None:
            with wave.open(args.wav) as w:
                frame = SAMPLE_RATE * FRAME_MS // 1000
                while chunk := w.readframes(frame):
                    audio.put_nowait(chunk)
                    await asyncio.sleep(FRAME_MS / 1000)
            await asyncio.sleep(1)
            stop()

        async def send() -> None:
            if args.wav:
                source = contextlib.nullcontext()
                player = asyncio.create_task(play_wav())
            else:
                source = sd.RawInputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="int16",
                    blocksize=SAMPLE_RATE * FRAME_MS // 1000,
                    device=args.device,
                    callback=on_audio,
                )
            with source:
                while (chunk := await audio.get()) is not None:
                    await stream.input_stream.send(_payload(chunk, "BINARY"))
            if args.wav:
                player.cancel()
            # Ask the server to flush the last turn and end the session.
            await stream.input_stream.send(_payload(b'{"type": "Terminate"}', "UTF8"))

        sender = asyncio.create_task(send())
        status = 0
        terminated = False
        last_final = None
        try:
            _, output = await stream.await_output()
            while True:
                # Once we've stopped sending, don't wait forever for the server.
                timeout = DRAIN_TIMEOUT_S if stopping.is_set() else None
                try:
                    event = await asyncio.wait_for(output.receive(), timeout)
                except TimeoutError:
                    _say(f"{RED}No reply from the server after stopping.{RESET}")
                    status = 1
                    break
                value = getattr(event, "value", None)
                if value is None or not hasattr(value, "bytes_"):
                    break  # the stream ended (or an event we don't know)
                msg = json.loads(value.bytes_)
                kind = msg.get("type")
                if kind == "Begin":
                    _say(f"{BOLD}Listening.{RESET} Speak; Ctrl+C to stop.\n")
                elif kind == "Turn":
                    text = msg.get("transcript", "")
                    if msg.get("end_of_turn"):
                        if msg.get("turn_order") != last_final:
                            _say(text)
                            last_final = msg.get("turn_order")
                    elif text:
                        _show_partial(text)
                elif kind == "Termination":
                    terminated = True
                    secs = msg.get("audio_duration_seconds")
                    _say(f"\n{GREY}Session ended ({secs} s of audio).{RESET}")
                    break
                elif kind == "Error":
                    _say(f"{RED}Error {msg.get('error_code')}: {msg.get('error')}{RESET}")
                    status = 1
        except Exception as e:  # noqa: BLE001 - surface whatever the SDK raises
            _say(f"{RED}{type(e).__name__}: {e}{RESET}")
            status = 1
        finally:
            stop()
            if not terminated and status == 0:
                _say(f"{RED}The stream ended without a Termination message.{RESET}")
                status = 1
            if not terminated:
                _say(
                    f"{GREY}Check the endpoint name and that it is InService; its container "
                    f"logs are in CloudWatch /aws/sagemaker/Endpoints/{args.endpoint}{RESET}"
                )
            # The SDK's send and close can block for good once the server has
            # gone, so give each a moment and move on.
            await _bounded(asyncio.shield(sender), CLOSE_TIMEOUT_S)
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            await _bounded(stream.close(), CLOSE_TIMEOUT_S)
    return status


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("endpoint", nargs="?")
    ap.add_argument("--profile", default=None, help="AWS profile (default: AWS_PROFILE / default chain)")
    ap.add_argument("--region", default=None, help="default: the profile's region")
    ap.add_argument("--query", default="", help="extra connection parameters, e.g. 'keyterms_prompt=[\"AssemblyAI\"]'")
    ap.add_argument("--device", type=int, default=None, help="input device index (see --list-devices)")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--wav", default=None, help="stream a 16 kHz mono 16-bit WAV instead of the mic")
    args = ap.parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return
    if not args.endpoint:
        ap.error("endpoint is required")
    if args.wav:
        with wave.open(args.wav) as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
                ap.error("--wav must be 16 kHz mono 16-bit PCM")
    try:
        sys.exit(asyncio.run(run(args)))
    except KeyboardInterrupt:  # a second Ctrl+C
        sys.exit(130)


if __name__ == "__main__":
    main()

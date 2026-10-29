#!/usr/bin/env python3
"""Transcribe many audio files from S3 with a SageMaker batch transform job.

Each S3 object is sent to the model as one request and its transcript is written
to the output prefix as <filename>.out

Note: batch transform sends each object as a raw request body, so configuration
options cannot be passed. Output contains word text and confidence but NOT
timestamps. Use a real-time endpoint when word-level timings are required.

Usage:
    python invoke_batch.py --model <name> --input s3://bucket/in/ --output s3://bucket/out/
"""
import argparse
import time

import boto3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="SageMaker model name")
    p.add_argument("--input", required=True, help="s3:// prefix containing WAV files")
    p.add_argument("--output", required=True, help="s3:// prefix for transcripts")
    p.add_argument("--instance-type", default="ml.g5.xlarge")
    p.add_argument("--region", default=None)
    args = p.parse_args()

    sm = boto3.client("sagemaker", region_name=args.region)
    job_name = f"{args.model}-{int(time.time())}"

    sm.create_transform_job(
        TransformJobName=job_name,
        ModelName=args.model,
        MaxConcurrentTransforms=1,
        MaxPayloadInMB=40,
        TransformInput={
            "DataSource": {
                "S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": args.input}
            },
            "ContentType": "audio/wav",
            "CompressionType": "None",
            # One file == one request. Without this SageMaker may split the
            # binary payload mid-audio.
            "SplitType": "None",
        },
        TransformOutput={
            "S3OutputPath": args.output,
            "Accept": "application/json",
            "AssembleWith": "None",
        },
        TransformResources={
            "InstanceType": args.instance_type,
            "InstanceCount": 1,
        },
    )
    print(f"Started transform job: {job_name}")

    sm.get_waiter("transform_job_completed_or_stopped").wait(TransformJobName=job_name)
    d = sm.describe_transform_job(TransformJobName=job_name)
    print(f"Status: {d['TransformJobStatus']}")
    if d["TransformJobStatus"] == "Failed":
        raise SystemExit(d.get("FailureReason", ""))
    print(f"Transcripts written to {args.output}")


if __name__ == "__main__":
    main()

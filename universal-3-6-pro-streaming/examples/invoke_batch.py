#!/usr/bin/env python3
"""Transcribe many audio files from S3 with a SageMaker batch transform job.

Each S3 object is one request; its transcript is written to the output prefix
as <filename>.out.

Batch transform cannot set request headers, so to pass session options put
each file in a JSON envelope, {"audio": "<base64>", "config": {...}}, and run
the job with --content-type application/json.

Usage:
    python invoke_batch.py --model <model-name> --input s3://bucket/in/ --output s3://bucket/out/
"""
import argparse
import time

import boto3

# Batch hosts top out at NVIDIA driver 535. The container enables CUDA
# compatibility libraries itself, so the platform must leave them alone.
TRANSFORM_AMI_VERSION = "al2-ami-sagemaker-batch-gpu-535"
ENVIRONMENT = {"SAGEMAKER_CUDA_COMPAT_DISABLED": "true"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="SageMaker model name (created by deploy.py)")
    p.add_argument("--input", required=True, help="s3:// prefix of the audio files")
    p.add_argument("--output", required=True, help="s3:// prefix for the transcripts")
    p.add_argument("--content-type", default="audio/wav", help="audio/wav, or application/json for envelopes")
    p.add_argument("--instance-type", default="ml.g5.2xlarge")
    p.add_argument("--region", default=None)
    args = p.parse_args()

    sm = boto3.client("sagemaker", region_name=args.region)
    job_name = f"{args.model}-{int(time.time())}"

    sm.create_transform_job(
        TransformJobName=job_name,
        ModelName=args.model,
        # MaxConcurrentTransforms and MaxPayloadInMB are left to the
        # container's defaults (4 concurrent records of up to 25 MB).
        BatchStrategy="SingleRecord",
        # How long one record may take; raise it for long recordings (max 3600).
        ModelClientConfig={"InvocationsTimeoutInSeconds": 600, "InvocationsMaxRetries": 0},
        Environment=ENVIRONMENT,
        TransformInput={
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": args.input}},
            "ContentType": args.content_type,
            "CompressionType": "None",
            # One file is one record; never split audio.
            "SplitType": "None",
        },
        TransformOutput={"S3OutputPath": args.output, "Accept": "application/json", "AssembleWith": "None"},
        TransformResources={
            "InstanceType": args.instance_type,
            "InstanceCount": 1,
            "TransformAmiVersion": TRANSFORM_AMI_VERSION,
        },
    )
    print(f"Started transform job {job_name}")

    sm.get_waiter("transform_job_completed_or_stopped").wait(
        TransformJobName=job_name, WaiterConfig={"Delay": 30, "MaxAttempts": 120}
    )
    d = sm.describe_transform_job(TransformJobName=job_name)
    print(f"Status: {d['TransformJobStatus']}")
    if d["TransformJobStatus"] != "Completed":
        raise SystemExit(d.get("FailureReason", ""))
    print(f"Transcripts written to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Deploy Universal-3.6 Pro Streaming to a SageMaker real-time endpoint.

The endpoint serves both live streaming (InvokeEndpointWithBidirectionalStream)
and whole-file transcription (InvokeEndpoint).

Usage:
    python deploy.py --model-package-arn <arn> --role-arn <arn> [--instance-type ml.g6e.xlarge]
    python deploy.py --delete [--name universal-3-6-pro-streaming]
"""
import argparse
import time

import boto3

# Recommended. Provides NVIDIA driver 580 (CUDA 13), which the model runs on
# natively; on an older default driver the container falls back to CUDA
# compatibility libraries.
INFERENCE_AMI_VERSION = "al2023-ami-sagemaker-inference-gpu-4-1"


def delete(sm, name):
    for call, kwargs in (
        (sm.delete_endpoint, {"EndpointName": name}),
        (sm.delete_endpoint_config, {"EndpointConfigName": name}),
        (sm.delete_model, {"ModelName": name}),
    ):
        try:
            call(**kwargs)
            print(f"Deleted {call.__name__.removeprefix('delete_')}: {name}")
        except sm.exceptions.ClientError as e:
            print(f"Skipped {call.__name__.removeprefix('delete_')}: {e.response['Error']['Message']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-package-arn", help="from your AWS Marketplace subscription")
    p.add_argument("--role-arn", help="SageMaker execution role")
    p.add_argument("--instance-type", default="ml.g6e.xlarge",
                   help="ml.g6e.xlarge, ml.g6e.2xlarge, ml.g6e.4xlarge, ml.g7e.2xlarge or ml.g7e.4xlarge (one GPU each)")
    p.add_argument("--name", default="universal-3-6-pro-streaming")
    p.add_argument("--region", default=None)
    p.add_argument("--delete", action="store_true", help="delete the endpoint, its config and the model")
    args = p.parse_args()

    sm = boto3.client("sagemaker", region_name=args.region)
    if args.delete:
        delete(sm, args.name)
        return
    if not (args.model_package_arn and args.role_arn):
        p.error("--model-package-arn and --role-arn are required")

    sm.create_model(
        ModelName=args.name,
        PrimaryContainer={"ModelPackageName": args.model_package_arn},
        ExecutionRoleArn=args.role_arn,
        # Required for AWS Marketplace model packages.
        EnableNetworkIsolation=True,
    )
    sm.create_endpoint_config(
        EndpointConfigName=args.name,
        ProductionVariants=[{
            "VariantName": "primary",
            "ModelName": args.name,
            "InitialInstanceCount": 1,
            "InstanceType": args.instance_type,
            "InferenceAmiVersion": INFERENCE_AMI_VERSION,
            # The weights are copied to the instance and loaded onto the GPU
            # at startup; allow ample time for both.
            "ModelDataDownloadTimeoutInSeconds": 1800,
            "ContainerStartupHealthCheckTimeoutInSeconds": 1800,
        }],
    )
    sm.create_endpoint(EndpointName=args.name, EndpointConfigName=args.name)
    print(f"Creating endpoint {args.name} on {args.instance_type}...")

    t0 = time.time()
    while True:
        d = sm.describe_endpoint(EndpointName=args.name)
        status = d["EndpointStatus"]
        if status == "InService":
            print(f"InService after {time.time() - t0:.0f}s")
            return
        if status in ("Failed", "OutOfService"):
            raise SystemExit(f"{status}: {d.get('FailureReason', '')}")
        time.sleep(20)


if __name__ == "__main__":
    main()

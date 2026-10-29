#!/usr/bin/env python3
"""Deploy Universal 3.5 Pro Sync to a SageMaker real-time endpoint.

Usage:
    python deploy.py --model-package-arn <arn> --role-arn <arn> [--instance-type ml.g5.xlarge]
"""
import argparse
import time

import boto3

# Required. Provides NVIDIA driver 535 with CUDA-compat mounting disabled.
# Without this the container exits during startup with CUDA error 803
# ("system has unsupported display driver / cuda driver combination").
INFERENCE_AMI_VERSION = "al2-ami-sagemaker-inference-gpu-2-1"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-package-arn", required=True)
    p.add_argument("--role-arn", required=True, help="SageMaker execution role")
    p.add_argument("--instance-type", default="ml.g5.xlarge")
    p.add_argument("--name", default="universal-3-5-pro-sync")
    p.add_argument("--region", default=None)
    args = p.parse_args()

    sm = boto3.client("sagemaker", region_name=args.region)

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
            # Weights load onto the GPU at startup; allow ample time.
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

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel


# IMPORTANT: YOU NEED TO CHANGE THESE
# -----------------------------
ROLE_ARN = "your-sagemaker-role-here" # specifically this one
BUCKET = "your-s3-bucket-name-here"   # and this one
PREFIX = "model-artifacts/spam-ham-svc"
ENDPOINT_NAME = "spam-ham-endpoint"


# SETUP
# -----------------------------
boto_session = boto3.Session()
region = boto_session.region_name
sm_session = sagemaker.Session(boto_session=boto_session)

# Upload model artifact to S3
model_s3_uri = sm_session.upload_data(
    path="model_build/model.tar.gz",
    bucket=BUCKET,
    key_prefix=PREFIX
)

print("Uploaded model to:", model_s3_uri)
print("Region:", region)

# Create SageMaker model object
sk_model = SKLearnModel(
    model_data=model_s3_uri,
    role=ROLE_ARN,
    entry_point="inference.py",
    source_dir="model_build",
    framework_version="1.4-2",
    py_version="py3",
    sagemaker_session=sm_session
)

# Deploy endpoint
predictor = sk_model.deploy(
    endpoint_name=ENDPOINT_NAME,
    instance_type="ml.m5.large",
    initial_instance_count=1
)

print("Endpoint deployed:", ENDPOINT_NAME)

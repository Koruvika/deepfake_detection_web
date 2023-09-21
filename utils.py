import boto3

# Configure AWS credentials
aws_access_key_id = 'AKIA2PZGTZ4CWGXVER4J'
aws_secret_access_key = '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'

# Configure the S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

bucket_name = 'deepfake-medias'

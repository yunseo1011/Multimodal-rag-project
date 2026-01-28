import boto3
import os
from botocore.exceptions import NoCredentialsError
from io import BytesIO
from datetime import datetime

class S3Client:
    def __init__(self):
        # 환경 변수에서 키를 가져옵니다 (도커가 주입해줌)
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'ap-northeast-2')
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME')

    def upload_file(self, file_obj, filename: str, session_id: str) -> str:
        """
        파일 객체(Bytes)를 받아 S3에 업로드하고 Key(경로)를 반환
        구조: uploads/{session_id}/{timestamp}_{filename}
        """
        try:
            # 1. 덮어쓰기 방지
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"uploads/{session_id}/{timestamp}_{filename}"

            # 2. 업로드 (upload_fileobj는 메모리에 있는 파일을 바로 올림)
            # file_obj.seek(0)은 파일 포인터를 맨 앞으로 돌리는 안전장치
            file_obj.seek(0)
            self.s3.upload_fileobj(file_obj, self.bucket_name, s3_key)
            
            print(f"✅ S3 Upload Success: {s3_key}")
            return s3_key
        
        except NoCredentialsError:
            print("❌ AWS 자격 증명이 없습니다.")
            raise
        except Exception as e:
            print(f"❌ S3 Upload Error: {e}")
            raise

    def get_file(self, s3_key: str) -> BytesIO:
        """
        S3 Key를 주면 파일 내용을 메모리(BytesIO)로 다운로드해 반환
        """
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=s3_key)
            file_content = response['Body'].read()
            return BytesIO(file_content)
        except Exception as e:
            print(f"❌ S3 Download Error: {e}")
            raise
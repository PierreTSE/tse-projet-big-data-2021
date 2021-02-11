import boto3
import botocore
import paramiko
import os
import configparser

os.path.dirname(os.path.abspath(__file__))
config = configparser.RawConfigParser()
config.read('properties.txt')

KEY=config.get('AWS','aws_access_key_id')
SECRET=config.get('AWS','aws_secret_access_key')
Token=config.get('AWS','aws_session_token')
bucket_name=config.get('AWS','bucket_name')
dossier_local_data=config.get('AWS','dossier_local_data')
dossier_local_result=config.get('AWS','dossier_local_result')
username=config.get('AWS','username')
hostname=config.get('AWS','hostname')
paire_cle=config.get('AWS','paire_cle')

s3 = boto3.client('s3', aws_access_key_id=KEY,aws_secret_access_key=SECRET,aws_session_token=Token, region_name='us-east-1')


# Methode pour créer un bucket
def create_bucket(bucket_name):
    try:
        s3.create_bucket(Bucket=bucket_name)
    except:
        return False
    return True

# Methode pour transferer des fichiers en local à un bucket
def upload_file(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = file_name
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except:
        return False
    return True

# Methode pour transferer un dossier en local à un bucket
def uploadDirectory(path,bucketname):
        for root,dirs,files in os.walk(path):
            for file in files:
                s3.upload_file(os.path.join(root,file),bucketname, 'data/{}'.format(file))


create_bucket(bucket_name)
# Restriction des acces

response_public = s3.put_public_access_block(
        Bucket=bucket_name,
        PublicAccessBlockConfiguration={
            'BlockPublicAcls': True,
            'IgnorePublicAcls': True,
            'BlockPublicPolicy': True,
            'RestrictPublicBuckets': True
        },
    )
    

# Transfere du dossier en local au bucket
uploadDirectory(dossier_local_data,bucket_name)

# Connexion ssh a l'instance
key = paramiko.RSAKey.from_private_key_file(paire_cle)
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    # Transfere des données du bucket à l'instance
    client.connect(hostname=hostname, username=username, pkey=key)
    cmd="aws s3 cp s3://"+bucket_name+"/data /home/ec2-user/data --recursive"
    stdin, stdout, stderr = client.exec_command(cmd)
    print(stdout.read())
    client.close()
except Exception:
    print("ERREUR CONNEXION")


import configparser
import json
import os

import boto3
import pandas as pd
import paramiko
import pymongo

os.path.dirname(os.path.abspath(__file__))
config = configparser.RawConfigParser()
config.read('properties.txt')

KEY = config.get('AWS', 'aws_access_key_id')
SECRET = config.get('AWS', 'aws_secret_access_key')
Token = config.get('AWS', 'aws_session_token')
bucket_name = config.get('AWS', 'bucket_name')
dossier_local_data = config.get('AWS', 'dossier_local_data')
dossier_local_result = config.get('AWS', 'dossier_local_result')
username = config.get('AWS', 'username')
hostname = config.get('AWS', 'hostname')
paire_cle = config.get('AWS', 'paire_cle')

host_mongodb = config.get('Mongodb', 'host_mongodb')
username_mongodb = config.get('Mongodb', 'username_mongodb')
password_mongodb = config.get('Mongodb', 'password_mongodb')
db_mongodb = config.get('Mongodb', 'db_mongodb')
collection_mongodb = config.get('Mongodb', 'collection_mongodb')
file_to_import = config.get('Mongodb', 'file_to_import')


# Methode pour transferer données du bucket a un dossier en local
def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)


# Connexion ssh a l'instance
key = paramiko.RSAKey.from_private_key_file(paire_cle)
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    # Copie des donnees de l'instance vers le bucket
    client.connect(hostname=hostname, username=username, pkey=key)
    cmd = "aws s3 cp /home/ec2-user/result s3://" + bucket_name + "/result --recursive"
    stdin, stdout, stderr = client.exec_command(cmd)
    print(stdout.read())
    client.close()
except Exception:
    print("ERREUR CONNEXION")

s3 = boto3.resource('s3', aws_access_key_id=KEY, aws_secret_access_key=SECRET, aws_session_token=Token,
                    region_name='us-east-1')

download_s3_folder(bucket_name, "result", dossier_local_result)


# Envoi des documents dans une base de donnnées NoSQL MongoDB en local


def import_content(filepath):
    mng_client = pymongo.MongoClient('localhost', 27017)
    mng_db = mng_client[db_mongodb]
    collection_name = collection_mongodb
    db_cm = mng_db[collection_name]
    cdir = os.path.dirname(__file__)
    file_res = os.path.join(cdir, filepath)
    data = pd.read_csv(file_res, encoding='utf-8', na_filter=False)
    data_json = json.loads(data.to_json(orient='records'))
    db_cm.insert_many(data_json)
    print("Données transférées sur MongoDb")


import_content(file_to_import)

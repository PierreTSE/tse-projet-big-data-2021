import boto3
import botocore
import paramiko
import os
import configparser
import time

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


# Connexion ssh a l'instance
key = paramiko.RSAKey.from_private_key_file(paire_cle)
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    # Executer process
    client.connect(hostname=hostname, username=username, pkey=key)
    channel = client.invoke_shell()
    stdin = channel.makefile('wb')
    stdout = channel.makefile('rb')
    stdin.write('''
    sudo yum install python3 -y
    cd /home/ec2-user
    mkdir result
    cd data
    sudo pip3 install -r requirements.txt
    python3 ProjetBigData2021-Copie.py
    exit
    ''')
    print(stdout.read())

    stdout.close()
    stdin.close()
    client.close()
except Exception:
    print("ERREUR CONNEXION")


"""
sudo yum install python3 -y ; sudo pip3 install -r requirements.txt; cd /home/ec2-user; mkdir result ; cd data ; python3 ProjetBigData2021-Copie.py
"""




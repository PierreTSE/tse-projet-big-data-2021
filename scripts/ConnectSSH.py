import configparser
import os

import paramiko
from scp import SCPClient

from utils import progress

# changement du répertoire de travail pour s'assurer de lire le fichier de configuration
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#### Lecture du fichier de configuration
config = configparser.ConfigParser()
config.read("config.txt")
address = config['Hadoop']['address']
port = int(config['Hadoop']['port'])
username = config['Hadoop']['username']
pwd = config['Hadoop']['password']
source_hadoop_directory = config['Hadoop']['source_directory']
remote_directory = os.path.basename(os.path.normpath(source_hadoop_directory))
target_local_directory = os.path.normpath(config['Local']['target_local_directory'])
####

### Connexion SSH + récupération des fichiers depuis HDFS
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(address, port=port, username=username, password=pwd, allow_agent=False)
command = 'if [ -d ' + remote_directory + ' ] ; then echo "data exists" ; else hdfs dfs -get ' + source_hadoop_directory + "; fi"
stdin, stdout, stderr = ssh.exec_command(command)
exit_status = stdout.channel.recv_exit_status()
if exit_status != 0:
    raise RuntimeError("hdfs could not get data files")
####

#### On s'assure que le dossier de destination existe
if not os.path.exists(target_local_directory):
    os.makedirs(target_local_directory)
os.chdir(target_local_directory)
####

### Récupération des fichiers sur Hadoop en local
with SCPClient(ssh.get_transport(), progress=progress) as scp:
    for file in ["categories_string.csv",
                 "label.csv",
                 "data.json"]:
        if not os.path.exists(file):
            scp.get(remote_directory + "/" + file, local_path=os.getcwd())
####

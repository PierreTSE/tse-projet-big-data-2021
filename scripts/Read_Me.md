# Script infos
Ce document comporte les informations nécessaires pour pouvoir lancer les scripts fournis en annexe de ce projet.
Vous retrouverez au sein de ce document :
- L'ensemble des modules nécessaires avec python 3
- Les informations associées au fichier config et leur script associé.

## Installation de python3
Vous devez dans un premier temps installer python 3 sur les différentes machines d’où vous lancez les scripts contenus dans ce fichier.
Ensuite vous devrez installer les modules nécessaires selon le script à lancer, nous allons procéder à la description des différents scripts.

##ConnectSSH.py
Permet la récupération des données d’hadoop à votre machine en local
A installer au préalable:
pip install paramiko

Fichier config associé : config.txt
Pour le lancer: python3 ConnectSSH.py

## create_instance.py
Crée une instance sur le cloud Amazon AWS
A installer au préalable:
pip install boto3 
pip install configparser

Fichier config: properties.txt 
Pour le lancer: python3 create_instance.py

##download_ec2_s3_local.py
Récupère un fichier CSV contenant le résultat des données traitées + Stockage dans une base de données NoSql MongoDB
A installer au préalable:
pip install configparser
pip install paramiko
pip install python-csv
pip install pymongo
pip install pandas

Fichier config: properties.txt 
Pour le lancer: python3 download_ec2_s3_local.py

##process_ec2.py:
Exécute le code pour le traitement des données sur l’instance ec2
A installer au préalable
pip install boto3 
pip install paramiko
pip install configparser

Fichier config: properties.txt 
Pour le lancer: python3 process_ec2.py

##upload_local_s3_ec2.py:
Transfert des données sur l’instance en passant par un bucket
A installer au préalable
pip install boto3 
pip install paramiko
pip install configparser

Fichier config: properties.txt 
Pour le lancer: python3 upload_local_s3_ec2.py

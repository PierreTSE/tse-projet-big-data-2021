# TSE Projet Big Data 2021

## Équipe
Alexandre Fouquet\
Pierre Giraud\
Youness Boumahdi

## Traitement de données

Le notebook `Projet Big Data 2021.ipynb` présente le déroulement des recherches, traitements, entraînement effectués au cours de ce projet.

Les données, la plupart des modèles entraînés, ainsi que l'objet de prétraitement et les données prétraitées, sont disponibles [ici (143MB)](https://drive.google.com/file/d/1qKh9XeugQzGo4_W0gk657K_urcOBbaYF/view?usp=sharing).

## Documentation des Scripts
Cette partie comporte les informations nécessaires pour pouvoir lancer les scripts fournis avec ce projet.
Vous retrouverez au :
- l'ensemble des modules Python nécessaires
- les informations associées au fichier config.txt et leur script associé.

### Installation de Python
Vous devez dans un premier temps installer Python 3.8 sur les différentes machines où vous lancerez les scripts.

Ensuite, vous devrez installer les modules nécessaires selon le script à lancer.

### import_from_hadoop.py
Permet la récupération des données d’Hadoop à votre machine en local.

Requiert :
`pip install configparser paramiko scp`

**Fichier config** : config.txt

### create_instance.py
Crée une instance sur le cloud Amazon AWS.

Requiert :

`pip install boto3 configparser`

**Fichier config** : properties.txt

### process_ec2.py:
Exécute le code de traitement des données sur l’instance ec2.

Requiert :

`pip install boto3 configparser paramiko`

**Fichier config**: properties.txt 

Le traitement, exposé dans le fichier script_AWS.py, consiste en un pré-traitement basé sur TF-IDF, et l'entraînement d'un modèle de multi-classification SVM linéaire (macro-F1=0.74, fairness=3.26)

### upload_local_s3_ec2.py:
Transfère des données sur l’instance en passant par un bucket.

Requiert :

`pip install boto3 configparser paramiko`


**Fichier config**: properties.txt 

### download_ec2_s3_local.py
Récupère un fichier CSV contenant le résultat des données traitées, et les stocke dans une base de données MongoDB.

Requiert :

`pip install configparser pandas paramiko pymongo python-csv`

**Fichier config**: properties.txt

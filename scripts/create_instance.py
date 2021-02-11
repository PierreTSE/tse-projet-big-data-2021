import boto3
import botocore
import paramiko
import os
import configparser

os.path.dirname(os.path.abspath(__file__))
path_fichier_config='properties.txt'
config = configparser.RawConfigParser()
config.read(path_fichier_config)

KEY=config.get('AWS','aws_access_key_id')
SECRET=config.get('AWS','aws_secret_access_key')
Token=config.get('AWS','aws_session_token')
id_instance=config.get('AWS','id_instance')
arn_iam=config.get('AWS','arn_iam')
paire_cle=config.get('AWS','paire_cle')
security_group=config.get('AWS','security_group')
nom_paire_cle=os.path.splitext(os.path.basename(paire_cle))[0]
instance_name=config.get('AWS','instance_name')

ec2 = boto3.resource('ec2', aws_access_key_id=KEY,aws_secret_access_key=SECRET,aws_session_token=Token, region_name='us-east-1')

existe=False
instances = ec2.instances.all()
for instance in instances:
        if instance.tags!= None:
            for tag in instance.tags:
                if tag['Key'] == 'Name':
                    if tag['Value'] == instance_name:
                        print("ERROR: Instance existe déjà")
                        existe=True
                        
if existe==False :
    #Creation d'instance
    instances = ec2.create_instances(
         ImageId='ami-047a51fa27710816e',
         MinCount=1,
         MaxCount=1,
         InstanceType='t2.micro',
         SecurityGroupIds=[security_group],
         KeyName=nom_paire_cle,
         TagSpecifications=[
        {
            'ResourceType': 'instance',
          'Tags': [
            {
              'Key': 'Name',
              'Value': instance_name
            }
          ]
        },
      ]
     )
     
    instance_id = instances[0].instance_id
    # Update des infos dans le fichier properties
    config.set('AWS', 'id_instance', instance_id)
    with open("properties1.txt", "w") as fh:
        config.write(fh)
    os.remove(os.path.dirname(path_fichier_config)+"properties.txt")
    os.rename(os.path.dirname(path_fichier_config)+"properties1.txt", os.path.dirname(path_fichier_config)+"properties.txt")
    
    id_instance=config.get('AWS','id_instance')
    # Lance l'instance et update des infos dans le fichier properties
    for instance in instances:
        if instance.instance_id==id_instance:
            # Wait for the instance to enter the running state
            instance.wait_until_running()
            # Reload the instance attributes
            instance.load()
            print(instance.public_dns_name)
            config.set('AWS', 'hostname', instance.public_dns_name)
            with open("properties1.txt", "w") as fh:
                config.write(fh)
            os.remove(os.path.dirname(path_fichier_config)+"properties.txt")
            os.rename(os.path.dirname(path_fichier_config)+"properties1.txt", os.path.dirname(path_fichier_config)+"properties.txt")

    ec2_client = boto3.client('ec2', aws_access_key_id=KEY,aws_secret_access_key=SECRET,aws_session_token=Token, region_name='us-east-1')
    # Associe Iam à l'instance
    response = ec2_client.associate_iam_instance_profile(
        IamInstanceProfile={
            'Arn': arn_iam,
            'Name': 'Role_full_access_s3'
        },
        InstanceId= instance_id
    )

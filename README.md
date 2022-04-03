# README

conda environment name: TSCResNet
docker image name: tsc-resnet

main library version:
- python 3.8.2
- pytorch 1.10.0
- mlflow 1.22.0
- pip 21.2.4
- conda 4.10.3

## .zshrc

see .zshrc
    - defines conda w/python3
    - defines homebrew source

## homebrew 

brew update  
brew upgrade

## Conda

conda update conda

conda env create -f environment.yml

conda env remove -n dissertation

conda env update --file environment.yml --prune

conda activate TSCResNet

conda env export > environment_20220207.yml

## Docker

### Build

```
docker build -t tsc-resnet .
```

### Tests

```
python -m pytest
```

## Deployment Commands

### Local deployment (run the deployment app from Local env to launch resources in AWS)

```
docker build -t tsc-resnet . && docker run -e AWS_DEFAULT_REGION=us-gov-west-1 -e AWS_ACCESS_KEY_ID=<> -e AWS_SECRET_ACCESS_KEY=<> -e S3_CONFIG_URL=s3://<path to config>/app_config.yaml -e ENVIRONMENT=<DEV, TEST, or PROD> -e DEPLOYMENT=<CREATE or DELETE> tsc-resnet
```

### Push a Docker Image to AWS ECR

```
aws ecr get-login-password --region us-gov-west-1 | docker login --username AWS --password-stdin 649705058163.dkr.ecr.us-gov-west-1.amazonaws.com

docker build -t tsc-resnet .

docker tag tsc-resnet:<VERSION> 649705058163.dkr.ecr.us-gov-west-1.amazonaws.com/tsc-resnet:<VERSION>

docker push 649705058163.dkr.ecr.us-gov-west-1.amazonaws.com/tsc-resnet:<VERSION>
```

### Deploy Commands

```
aws ecr get-login-password --region us-gov-west-1 | docker login --username AWS --password-stdin 649705058163.dkr.ecr.us-gov-west-1.amazonaws.com

docker pull 649705058163.dkr.ecr.us-gov-west-1.amazonaws.com/tsc-resnet:<VERSION>

docker run --rm 649705058163.dkr.ecr.us-gov-west-1.amazonaws.com/tsc-resnet:<VERSION>
```

### AWS ECR/ECS deployment

```
docker run -d --rm tsc-resnet:<VERSION>
```

## Docker - Some useful commands
#### List & Remove all stopped containers
```
docker container ls
docker rm $(docker ps -a -q)
```

#### List & Remove all dangling images
```
docker images
docker image prune
```

#### Remove Specific Images
```
docker image rm <ImageName>
```  

#### Remove All Images
```
docker rmi $(docker images -a -q)
```  

## MLFLOW

mlflow ui --backend-store-uri /Users/dgonzalez/Documents/dissertation/mlruns/

## Git

```git config --list```

```git log```

```git add . && git commit -m "message" && git push```

```git branch -a```

### Github PAT update
Once the personal access token expires, regenrate the token on github.com, save it, attempt to push and a prompt should come up where you can enter the updated token.

### Creating New Branches
```
git branch <branch>
git checkout <branch>
```
or 
```git checkout -b <branch>```

### Push New Branch to Origin
```git push -u origin <branch>```

thereafter:
```git push```

### Deleting Branches
First switch to another branch

Delete local:
```git branch -d <branch>```

Delete remote
```git push origin --delete <branch>```


### Launch a Jupyter Notebook environment (needs Dockerfile modification)

```
#In Dockerfile, uncomment:
#ENTRYPOINT source activate <environment name: defined in environment.yml> && jupyter notebook --ip=0.0.0.0 --port=8889 --no-browser --allow-root
```


# README

main library version:
- python 3.8.2
- pytorch 1.10.0
- mlflow 1.22.0
- pip 21.2.4
- conda 4.10.3

## Conda

conda env remove -n dissertation

conda env create -f environment.yml

conda env update --file environment.yml --prune

## Docker

## MLFLOW

mlflow ui --backend-store-uri /Users/dgonzalez/Documents/dissertation/mlruns/

## Git
```git log```

```git add . && git commit -m "message" && git push```

```git branch -a```

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

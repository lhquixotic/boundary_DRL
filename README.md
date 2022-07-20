# Boundary DRL
An implementation of DRL autonomous driving in autonomous driving, based on [gym_carla environment](https://github.com/cjy1992/gym-carla).

## System Requirements
- Ubuntu 18.04

## Installation
1. Setup conda environment
```
$ conda create -n env_name python=3.7
$ conda activate env_name
```

2. Clone this git repo in an appropriate folder
```
$ git clone https://github.com/lhquixotic/boundary_DRL.git
```

3. Enter the repo root folder and install the packages:
```
$ pip install -r requirements.txt
$ pip install -e .
```

4. Download [CARLA_0.9.8](https://github.com/carla-simulator/carla/releases/tag/0.9.8), extract it to some folder, and add CARLA to ```PYTHONPATH``` environment variable:
```
$ export PYTHONPATH=$PYTHONPATH:$YourFolder$/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.8-py3.5-linux-x86_64.egg
```

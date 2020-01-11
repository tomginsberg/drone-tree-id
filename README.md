# drone-tree-id

Tree crown instance segmentation with Detectron2.

## Directory Structure

## Setup Env on Leonard

### login
Ensure you are connected to the ETH network.
```
ssh [your_ETH_alias]@login.leonhard.ethz.ch
```

### run setup script
```
chmod +x setup
./setup
```

Alternatively, you can manually setup your environment by modifying:
#### load python, c++ module
```
module load gcc/6.3.0 python_gpu/3.7.4
```

#### clone repo
```
cd ~
git clone https://github.com/roeetal/drone-tree-id.git
```

#### install dependencies
GDAL current not installing on Leonard.
```
python3 -m pip install --user rasterio pyshp opencv-python torch torchvision
```

#### install Detectron2 
If running on mac, you will need to specify the c++ compiler when installing dependencies: `MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ pip install ...`
```
cd ~/drone-tree-id
mkdir -p lib
git clone https://github.com/facebookresearch/detectron2.git lib/detectron2
python3 -m pip install --user -e lib/detectron2/.
```

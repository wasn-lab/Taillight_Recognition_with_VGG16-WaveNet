### How to use:

```
bash main.sh foo.bag [*.bag]
```

This command will automatically extract images, run deelap, yolo, etc
to detect objects, and compare the detection results.
In the end, the most tricky images will be selected and put in the same directory.


### Environment setup for Yet-Another-EfficientDet-Pytorch

Yet-Another-EfficientDet-Pytorch can run in pytorch 1.2.
Higher versions are OK if you want to use them, but you may need to
pick up the compatible torchvision.

```
readonly venv_dir=py36_efficientdet
pushd $HOME
mkdir -p $venv_dir
python3.6 -m venv $venv_dir
source ${venv_dir}/bin/activate
pip install wheel
pip install cython
pip install numpy
pip install pycocotools tqdm tensorboard tensorboardX pyyaml webcolors tensorflow-gpu==1.13.1
pip install torch==1.2.0
pip install torchvision==0.4.0
pip install matplotlib
popd
```

More if you need to run lanenet-lane-detection
```
pip install glog sklearn loguru
```


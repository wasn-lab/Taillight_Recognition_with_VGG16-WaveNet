sudo apt install -y virtualenv
virtualenv -p /usr/bin/python2.7 sandbox
source ./sandbox/bin/activate
pip install -r requirements.txt
pip install pycocotools==2.0.0
pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html

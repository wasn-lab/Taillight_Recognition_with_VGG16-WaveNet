sudo apt install -y virtualenv
virtualenv -p /usr/bin/python2.7 sandbox
source ./sandbox/bin/activate
wget https://sourceforge.net/projects/pyqt/files/sip/sip-4.19.7/sip-4.19.7.tar.gz/download
tar -xzvf download
cd sip-4.19.7
python configure.py
make
sudo make install
cd -
rm -r download sip-4.19.7
pip install -r requirements.txt
pip install pycocotools==2.0.0
pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html

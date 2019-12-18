## server example

sudo apt-get -y install nfs-kernel-server
sudo update-rc.d nfs-kernel-server enable
sudo bash -c "echo '/home/d300/h/h_sensing/src 192.168.0.0/24(rw,insecure,sync,no_subtree_check)' >> /etc/exports"
sudo exportfs -r
sudo /etc/init.d/nfs-kernel-server restart

## client example

sudo apt-get -y install autofs
sudo update-rc.d autofs enable
sudo bash -c "echo '/home/d300/h/h_sensing /etc/auto.nfs' >> /etc/auto.master"
sudo bash -c "touch /etc/auto.nfs"
sudo bash -c "echo 'src 192.168.0.1:/home/d300/h/h_sensing/src' >> /etc/auto.nfs"
sudo /etc/init.d/autofs restart
cd ~
sudo mkdir /h_sensing
sudo ln -s /home/d300/h/h_sensing/src /home/nvidia/h_sensing/src

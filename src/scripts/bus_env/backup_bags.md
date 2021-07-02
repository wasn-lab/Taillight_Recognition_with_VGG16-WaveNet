1.  中斷camera ipc和車內內網的連線
```
sudo ifconfig enp0s31f6 down
```

2. 確定已成功斷開車內內網
```
ping 192.168.1.200  # （車內的OBU IP) 
```
若無回應表示使用的是實驗室線路，可繼續下一步，
反之，若有回應表示仍使用車內網路，
要等到Network is Unreachable或Destination Host Unreachable才可以

3. 接上實驗室網路:
把車庫牆邊那條網路線插到camera ipc裡，此時會從dhcp取得的另一個.1網域的ip

4.  確定使用的是實驗室線路:
```
ping 192.168.1.3  # （實驗室裡Lidar IPC）
```
有回應後，再執行
```
arp -a
```
此時輸出應可看到
```
? (192.168.1.3) at 78:d0:04:2a:ad:50
```

5. 把bag上傳到nas裡

正常的上傳速度應是在100MB/s上下，以下是使用lftp的例子

```
lftp -u your_name nas.itriadv.co
> cd /bag/竹北HSR/B1
> mirror -R bag_dir_name
```

#!/usr/bin/env python
import socket
import json
import time
addr=('192.168.43.204',8766)
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
count = 0
while 1:
    data = {"SPaT_MAP_Info":{"Latitude": 123.456,  "Longitude":456.789, "Spat_state" : 101, "Spat_sec": 15.5, "Signal_state":0, "Index": count} }
    if not data:
        break
    s.sendto(json.dumps(data),addr)
    count += 1
    time.sleep(1) #delay 1 sec
s.close()

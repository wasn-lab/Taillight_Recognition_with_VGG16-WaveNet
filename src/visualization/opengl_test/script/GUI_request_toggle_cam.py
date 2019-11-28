import requests
import json

# IP = '192.168.50.127'
# IP = '192.168.43.77'
# IP = '140.96.109.216'
IP = '0.0.0.0'
port = '6060'

# payload = {'data1':1, 'data2':'2', 'data3':[1,2,3], 'data4':None}
payload = dict()
payload["gui_name"] = "passenger" # "driver" # "passenger"
payload["cam_view_mode"] = "toggle"
payload["image3D"] = "on"
payload["image_surr"] = "on"
payload["cam_op"] = "" # "reset"
print("payload(json): " + json.dumps(payload))
# r = requests.get('http://%s:%s/json_in_data' % (IP, port), params={'data':json.dumps(payload)})
r = requests.post('http://%s:%s/json_in_out' % (IP, port), json=payload)

# T = str(r.text)
T = r.text
print(T)
try:
    JT = json.loads(T)
    print(JT)
except:
    ptint("No json received.")



J = r.json()

print(J)

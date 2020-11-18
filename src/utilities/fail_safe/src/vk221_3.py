"""
API for notifying that we have a new bag to be uploaded to backend.
"""
import os
import requests
from sb_param_utils import get_license_plate_number
from backend_info import WEBAPI_BASE_URL

def build_vk221_3_url(bag, plate=None):
    if plate is None:
        plate = get_license_plate_number()
    _, base = os.path.split(bag)
    return ("{}/WebAPI?type=M8.2.VK221_3"
            "&plate={}&bag_file={}".format(WEBAPI_BASE_URL, plate, base))


def notify_backend_with_new_bag(bag, plate=None):
    url = build_vk221_3_url(bag, plate)
    resp = requests.post(url)
    return resp.json()

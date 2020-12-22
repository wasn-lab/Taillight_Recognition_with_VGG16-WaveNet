# -*- encoding: utf-8 -*-
"""
API for notifying that a bag has been successfully uploaded to backend.
"""
import os
import sys
import requests
from sb_param_utils import get_license_plate_number
from backend_info import WEBAPI_BASE_URL

def build_vk221_4_url(bag, plate=None):
    if plate is None:
        plate = get_license_plate_number()
    _, base = os.path.split(bag)
    return (u"{}/WebAPI?type=M8.2.VK221_4"
            u"&plate={}&bag_file={}".format(WEBAPI_BASE_URL, plate, base))


def notify_backend_with_uploaded_bag(bag, plate=None):
    url = build_vk221_4_url(bag, plate)
    if sys.version_info.major == 2:
        resp = requests.post(url.decode("utf-8"))
    else:
        resp = requests.post(url)
    return resp.json()

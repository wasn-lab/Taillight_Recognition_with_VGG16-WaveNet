# -*- encoding: utf-8 -*-
import unittest
from msgs.msg import DetectedObjectArray, DetectedObject, CamInfo
from status_level import OK, WARN
from heartbeat import cam_object_detection_func

def _gen_detected_object():
    obj = DetectedObject()
    obj.bPoint.p0.x = 3
    obj.bPoint.p0.y = 2
    obj.bPoint.p0.z = 0
    obj.bPoint.p1.x = 3
    obj.bPoint.p1.y = 2
    obj.bPoint.p1.z = 1.7
    obj.bPoint.p2.x = 3.5
    obj.bPoint.p2.y = 2
    obj.bPoint.p2.z = 1.7
    obj.bPoint.p3.x = 3.5
    obj.bPoint.p3.y = 2
    obj.bPoint.p3.z = 0
    obj.bPoint.p4.x = 3
    obj.bPoint.p4.y = 2.5
    obj.bPoint.p4.z = 0
    obj.bPoint.p5.x = 3
    obj.bPoint.p5.y = 2.5
    obj.bPoint.p5.z = 1.7
    obj.bPoint.p6.x = 3.5
    obj.bPoint.p6.y = 2.5
    obj.bPoint.p6.z = 1.7
    obj.bPoint.p6.x = 3.5
    obj.bPoint.p6.y = 2.5
    obj.bPoint.p6.z = 0
    return obj

def _gen_cam_info_array(low_prob=False):
    ret = []
    for i in range(8):
        cam_info = CamInfo()
        cam_info.id = i
        if not low_prob:
            cam_info.prob = (i+1) * 0.1
        else:
            cam_info.prob = (i+1) * 0.01
        ret.append(cam_info)
    return ret



class HeartbeatTest(unittest.TestCase):
    def test_cam_object_detection_func(self):
        obj = _gen_detected_object()
        obj.camInfo = _gen_cam_info_array(low_prob=False)

        det_obj_arr = DetectedObjectArray()
        det_obj_arr.objects.append(obj)

        status, status_str = cam_object_detection_func(det_obj_arr)
        self.assertEqual(status, OK)
        self.assertEqual(status_str, "OK")

        obj.camInfo = _gen_cam_info_array(low_prob=True)
        status, status_str = cam_object_detection_func(det_obj_arr)
        self.assertEqual(status, WARN)
        self.assertTrue("Low confidenc" in status_str)

if __name__ == "__main__":
    unittest.main()

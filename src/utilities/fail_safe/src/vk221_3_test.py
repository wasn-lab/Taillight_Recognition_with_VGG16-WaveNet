import unittest
from vk221_3 import build_vk221_3_url, notify_backend_with_new_bag

class VK221_3Test(unittest.TestCase):
    def test_build_url(self):
        bag = "auto_record_2020-10-06-16-24-34_18.bag"
        plate = "BMW-5678"
        expect = ("http://60.250.196.127:8008/WebAPI?type=M8.2.VK221_3"
                  "&plate=BMW-5678"
                  "&bag_file=auto_record_2020-10-06-16-24-34_18.bag")
        self.assertEqual(build_vk221_3_url(bag, plate), expect)

        expect = ("http://60.250.196.127:8008/WebAPI?type=M8.2.VK221_3"
                  "&plate=ITRI-ADV"
                  "&bag_file=auto_record_2020-10-06-16-24-34_18.bag")
        self.assertEqual(build_vk221_3_url(bag), expect)

    @unittest.skip("No real invocation of web api in daily operations")
    def test_notify_backend_with_new_bag(self):
        bag = "auto_record_2020-10-06-16-24-34_18.bag"
        resp_json = notify_backend_with_new_bag(bag)
        self.assertEqual(resp_json["messageObj"]["msgCode"], 200)

if __name__ == "__main__":
    unittest.main()

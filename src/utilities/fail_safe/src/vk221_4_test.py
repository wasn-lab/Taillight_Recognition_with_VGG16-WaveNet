import unittest
from vk221_4 import build_vk221_4_url, notify_backend_with_uploaded_bag

class VK221_3Test(unittest.TestCase):
    def test_build_url(self):
        bag = "auto_record_2020-10-06-16-24-34_18.bag"
        plate = "BMW-5678"
        expect = ("http://60.250.196.127:8008/WebAPI?type=M8.2.VK221_4"
                  "&plate=BMW-5678"
                  "&bag_file=auto_record_2020-10-06-16-24-34_18.bag")
        self.assertEqual(build_vk221_4_url(bag, plate), expect)

        expect = ("http://60.250.196.127:8008/WebAPI?type=M8.2.VK221_4"
                  "&plate=ITRI-ADV"
                  "&bag_file=auto_record_2020-10-06-16-24-34_18.bag")
        self.assertEqual(build_vk221_4_url(bag), expect)

    @unittest.skip("No real invocation of web api in daily operations")
    def test_notify_backend_with_uploaded_bag(self):
        bag = "auto_record_2020-10-06-16-24-34_18.bag"
        resp_json = notify_backend_with_uploaded_bag(bag)
        self.assertEqual(resp_json["messageObj"]["msgCode"], 200)

if __name__ == "__main__":
    unittest.main()

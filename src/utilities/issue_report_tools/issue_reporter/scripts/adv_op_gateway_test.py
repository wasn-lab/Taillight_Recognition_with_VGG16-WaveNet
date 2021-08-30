import unittest
from adv_op_gateway import parse_advop_req_run_stop

class ADVOpTest(unittest.TestCase):
    def test_parse_advop_req_run_stop(self):
        payload = '{"state":1,"timestamp":1628832738652}'
        self.assertTrue(parse_advop_req_run_stop(payload))

        payload = '{"state":0,"timestamp":1628832738652}'
        self.assertFalse(parse_advop_req_run_stop(payload))

        payload = '1'
        self.assertTrue(parse_advop_req_run_stop(payload))

        payload = '0'
        self.assertFalse(parse_advop_req_run_stop(payload))

if __name__ == "__main__":
    unittest.main()

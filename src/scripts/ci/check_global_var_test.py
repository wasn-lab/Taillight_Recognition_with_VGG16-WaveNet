#!/usr/bin/env python
import unittest
import check_global_var


class CheckGlobalVarNamingTest(unittest.TestCase):
    def test_parse_var_decl_0(self):
        decl = "|-VarDecl 0x7f1d1085ab28 <line:23:1, line:26:1> line:23:24 used cam_ids_ 'const std::vector<int>':'const std::vector<int, std::allocator<int> >' listinit"
        ret = check_global_var._parse_var_decl(decl)
        self.assertEqual(ret["var"], "cam_ids_")
        self.assertEqual(ret["line"], "23")
        self.assertEqual(ret["decl_type"], "const std::vector<int>")

    def test_parse_var_decl_1(self):
        decl = "|-VarDecl 0x7f1fbf45b278 <col:1, col:23> col:23 used distCoeffs 'cv::Mat':'cv::Mat' callinit"
        ret = check_global_var._parse_var_decl(decl)
        self.assertEqual(ret["var"], "distCoeffs")
        self.assertEqual(ret["line"], "undetected")
        self.assertEqual(ret["decl_type"], "cv::Mat")

    def test_parse_var_decl_2(self):
        decl = "|-VarDecl 0x7ff647772270 <line:93:1, col:30> col:5 used rawimg_size 'int' cinit"
        ret = check_global_var._parse_var_decl(decl)
        self.assertEqual(ret["var"], "rawimg_size")
        self.assertEqual(ret["line"], "93")
        self.assertEqual(ret["decl_type"], "int")

    def test_parse_var_decl_3(self):
        decl = "|-VarDecl 0x7f272e88e478 <line:54:1, col:6> col:6 used isInferData_3 'bool'"
        ret = check_global_var._parse_var_decl(decl)
        self.assertEqual(ret["var"], "isInferData_3")
        self.assertEqual(ret["line"], "54")
        self.assertEqual(ret["decl_type"], "bool")

    def test_parse_var_decl_4(self):
        decl = "|-VarDecl 0xe1c1740 <line:65:1, col:14> col:14 mutex_serverStatus 'boost::mutex':'boost::mutex' callinit"
        ret = check_global_var._parse_var_decl(decl)
        self.assertEqual(ret["var"], "mutex_serverStatus")
        self.assertEqual(ret["line"], "65")
        self.assertEqual(ret["decl_type"], "boost::mutex")

    def test_is_global_var_naming(self):
        self.assertFalse(check_global_var._is_global_var_naming("distCoeffs"))
        self.assertFalse(check_global_var._is_global_var_naming("rawimg_size"))
        self.assertFalse(check_global_var._is_global_var_naming("isInferData_3"))
        self.assertFalse(check_global_var._is_global_var_naming("g_isInferData_3"))
        self.assertTrue(check_global_var._is_global_var_naming("g_rawimg_size"))

#    @unittest.skip(True)
    def test_check_cpp_global_var_naming(self):
        check_global_var.check_cpp_global_var_naming(
            "src/sensing/itri_drivenet/drivenet/src/drivenet_120_1_b1.cpp")


if __name__ == "__main__":
    unittest.main()

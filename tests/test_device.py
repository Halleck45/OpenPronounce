import unittest
from unittest.mock import patch

import torch

from openpronounce import device


class TestDevice(unittest.TestCase):

    def setUp(self):
        device.get_device.cache_clear()

    def tearDown(self):
        device.get_device.cache_clear()

    def test_env_wins(self):
        with patch.dict("os.environ", {"OPENPRONOUNCE_DEVICE": "cpu"}):
            self.assertEqual(device.get_device(), torch.device("cpu"))

    def test_default_is_cuda_when_available(self):
        with patch.dict("os.environ", {}, clear=False), patch("torch.cuda.is_available", return_value=True):
            import os
            os.environ.pop("OPENPRONOUNCE_DEVICE", None)
            self.assertEqual(device.get_device().type, "cuda")

    def test_default_is_cpu_otherwise(self):
        with patch.dict("os.environ", {}, clear=False), patch("torch.cuda.is_available", return_value=False):
            import os
            os.environ.pop("OPENPRONOUNCE_DEVICE", None)
            self.assertEqual(device.get_device(), torch.device("cpu"))

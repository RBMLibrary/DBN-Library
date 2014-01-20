import unittest
from dbnlib.dbn import DBN

class TestDBN(unittest.TestCase):
    def test_constructor(self):
        d = DBN()
        self.assertEquals(True, True)



import unittest
from app.kohonen import Kohonen

class TestKohonen(unittest.TestCase):

    def test_train(self):
        self.assertEqual(Kohonen.train(),)
        self.assertIsInstance(Kohonen.train(),)

    def test_BMU_calculator(self):
        pass

    def test_update_weights_nbrs(self):
        pass

if __name__ == '__main__':
    unittest.main()
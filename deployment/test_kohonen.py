import unittest
import numpy as np
from app.kohonen import Kohonen

class TestKohonen(unittest.TestCase):

    def setUp(self):
        self.som1 = Kohonen(4,4)
        self.V = np.array([0.38767448, 0.11202134, 0.03882885])
        self.node_weights=np.array([[0.00396364, 0.61519785, 0.13264597],
                                [0.76098404, 0.53324604, 0.29676382],
                                [0.83614883, 0.03194745, 0.6797193 ],
                                [0.68718039, 0.78026006, 0.1371121 ],
                                [0.49024304, 0.83107034, 0.43617776],
                                [0.7973575 , 0.82200165, 0.48282774],
                                [0.10153806, 0.04857261, 0.94577383],
                                [0.81182771, 0.83797672, 0.44799182],
                                [0.47749725, 0.8168692 , 0.06920892],
                                [0.6813134 , 0.42145836, 0.82426671],
                                [0.10569722, 0.0297924 , 0.68569143],
                                [0.85896724, 0.65099824, 0.70085052],
                                [0.77259348, 0.14497027, 0.65211652],
                                [0.62709678, 0.47918591, 0.62487688],
                                [0.20917848, 0.20833123, 0.98215476],
                                [0.09258102, 0.94836982, 0.07684902]])
    
    def test_BMU_calculator(self):
        self.assertEqual(self.som1.BMU_calculator(V=self.V, node_weights=self.node_weights),1)
        self.assertIsInstance(self.som1.BMU_calculator(V=self.V, node_weights=self.node_weights),np.int64)

    def test_get_nbr_nodes(self):
        pass
    
    def test_update_weights_nbrs(self):
        pass
    
    def test_train(self):
        pass


if __name__ == '__main__':
    unittest.main()
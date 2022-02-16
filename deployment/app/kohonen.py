import numpy as np

class kohonen():
    
    def __init__(self, sizeX, sizeY):
        self.sizeX, self.sizeY = sizeX, sizeY
        self.num_nodes = sizeX*sizeY
        self.node_coords = np.array([[x,y] for y in range(0, self.sizeY) for x in range(0,self.sizeX)])
        
    def train(self, input_data, nmax_iter):
        ''' Train the SOM with the input training data'''
        node_weights = np.random.random((self.num_nodes, input_data.shape[1]))

        init_rad = max(self.sizeX, self.sizeY) / 2
        time_constant = nmax_iter / np.log(init_rad)
        
        for t in range(1, nmax_iter+1):
            for idx, V in enumerate(input_data):
                bmu_idx = self.BMU_calculator(V, node_weights)
                nbr_rad = init_rad*np.exp(-t / time_constant)
                nbr_nodes = self.get_nbr_nodes(bmu_idx, nbr_rad)
                node_weights = self.update_weights_nbrs(t, bmu_idx, V, nbr_nodes, node_weights, time_constant, nbr_rad)

        return node_weights
        
    
    def BMU_calculator(self, V, node_weights):
        '''Returns index of node in node weights that is the BMU'''
        euc_distances = np.sqrt(np.sum((V - node_weights)**2,axis=1))
        return np.where(euc_distances == np.min(euc_distances))[0][0]
        
    def get_nbr_nodes(self, bmu_idx, nbr_rad):
        '''Return list of indexes of nodes that fall within neighborhood radius of BMU'''
        euc_dis = np.sqrt(np.sum((self.node_coords - self.node_coords[bmu_idx])**2, axis=1))
        return list(np.where(euc_dis <= nbr_rad)[0])
    
    def update_weights_nbrs(self, t, bmu_idx, V, nbr_nodes, node_weights, time_constant, nbr_rad):
        ''' Returns updated weights for all nodes based on proximity to BMU'''
        learning_rate = 0.1*np.exp(-t/time_constant)
        nbr_node_weights = node_weights[nbr_nodes]
        nbr_node_coords, bmu_coords  = self.node_coords[nbr_nodes], self.node_coords[bmu_idx]

        euc_dis = np.sqrt(np.sum((bmu_coords - nbr_node_coords)**2,axis=1))
        influences = np.exp(-euc_dis**2 / (2*nbr_rad**2)) 
        node_weights[nbr_nodes] = nbr_node_weights + learning_rate*(influences*(V - nbr_node_weights).T).T 
        return node_weights
    
    def __str__(self):
        return f'Kohonen network of shape ({self.sizeX},{self.sizeY})'
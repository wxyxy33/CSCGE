import numpy as np
import scipy.sparse as sp
import torch
import json
import networkx as nx
import os

class Dataset:
    def __init__(self, args, path, dataset_name):
        self.dataset_root = path
        self.dataset_name = dataset_name
        self.dataset_path = path + '/' + dataset_name + '/'
        self.adj = None
        self.features = None
        self.labels = None
        self.Adj_dense = None
        self.work_path = args.work_path
        
    
    def preprogress(self, no_attributes=False, random_Attributes_dim=None):
        
        print('Loading [{}] dataset...'.format(self.dataset_name))
        
        if self.dataset_name == 'cora' or self.dataset_name == 'citeseer':
            idx_features_labels = np.genfromtxt("{}{}.content".format(self.dataset_path, self.dataset_name), dtype=np.dtype(str))
            edges_undirect = np.genfromtxt("{}{}.cites".format(self.dataset_path, self.dataset_name), dtype=np.dtype(str))
            
            id = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
            labels = self.encode_onehot(idx_features_labels[:, -1])
            idMapidx = {j: i for i, j in enumerate(id)}
            
            edges = np.array(list(map(idMapidx.get, edges_undirect.flatten())),
                                dtype=np.int32).reshape(edges_undirect.shape)
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(labels.shape[0], labels.shape[0]),
                                dtype=np.float32)

            if no_attributes:
                nodes_num = len(idx_features_labels)
                if random_Attributes_dim:
                    print('use random init feature!')
                    node_indices = np.random.permutation(np.arange(nodes_num))
                    features = sp.coo_matrix((np.ones(nodes_num), (np.arange(nodes_num), node_indices)), shape=(nodes_num, random_Attributes_dim))
                else:
                    print('use graph structure init feature!')
                    features = sp.eye(nodes_num)
                    features = adj + sp.eye(adj.shape[0])
            else:
                print('use origin feature!')
                features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
            features = self.normalize(features)
            features = torch.FloatTensor(np.array(features.todense()))
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj = self.normalize(adj + sp.eye(adj.shape[0]))
            Adj_dense = self.sparse_mx_to_torch_sparse_tensor(adj).to_dense()
            adj = self.sparse_mx_to_torch_sparse_tensor(adj)
            
            labels = torch.LongTensor(np.where(labels)[1])
            
            self.adj = adj
            self.features = features
            self.labels = labels
            self.Adj_dense = Adj_dense
            self.save()
            
            return adj, features, labels, Adj_dense
        
        elif self.dataset_name == 'deezer' or self.dataset_name == 'facebook_PPN':
            if self.dataset_name == 'deezer':
                feature_path = self.dataset_path + '/' + 'deezer_europe_features.json'
                edges_path = self.dataset_path + '/' + 'deezer_europe_edges.csv'
            elif self.dataset_name == 'facebook_PPN':
                feature_path = self.dataset_path + '/' + 'musae_facebook_features.json'
                edges_path = self.dataset_path + '/' + 'musae_facebook_edges.csv'
            
            with open(feature_path, "r") as f:
                id_feature = json.load(f)
            id_feature = {int(k):v for k,v in id_feature.items()}   
            idMapidx = {j: i for i, j  in enumerate(id_feature.keys())}
            edges_undirect = []
            
            with open(edges_path, 'r') as fp:
                for edge in fp:
                    edge= edge.strip().split(',')
                    edge = [int(i) for i in edge]
                    edges_undirect.append(edge)
  
            edges_undirect = np.array(edges_undirect)
            edges = np.array(list(map(idMapidx.get, edges_undirect.flatten())),
                                dtype=np.int32).reshape(edges_undirect.shape)
            
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(len(id_feature), len(id_feature)),
                                dtype=np.float32)
            
            if no_attributes:
                nodes_num = len(id_feature)
                if random_Attributes_dim:  
                    print('use random init feature!')
                    node_indices = np.random.permutation(np.arange(nodes_num))
                    features = sp.coo_matrix((np.ones(nodes_num), (np.arange(nodes_num), node_indices)), shape=(nodes_num, random_Attributes_dim))
                else:
                    print('use graph structure init feature!')
                    features = sp.eye(nodes_num)
                    features = adj + sp.eye(adj.shape[0])
            else:
                print('use origin feature!')
                pass
            
            features = self.normalize(features)
            features = torch.FloatTensor(np.array(features.todense()))
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj = self.normalize(adj + sp.eye(adj.shape[0]))
            Adj_dense = self.sparse_mx_to_torch_sparse_tensor(adj).to_dense()
            adj = self.sparse_mx_to_torch_sparse_tensor(adj)
            
            labels = torch.ones((1, 1))
            # features = torch.rand(28281, 2048)
            
            self.adj = adj
            self.features = features
            self.Adj_dense = Adj_dense
            
            self.save()
            
            return adj, features, None, Adj_dense
            
        elif self.dataset_name == 'ppi':
        
            edges_path = self.dataset_path + '/' + 'ppi-walks.txt'
            feature_path = None
            nodes = [] 
            edges_undirect = []
            with open(edges_path, 'r') as fp:
                for edge in fp:
                    edge= edge.strip().split('\t')
                    edge = [int(i) for i in edge]
                    edges_undirect.append(edge)
                    nodes.extend(edge)
            nodes = list(set(nodes))
            
            idMapidx = {j:i for i,j in enumerate(nodes)}

            edges_undirect = np.array(edges_undirect)
            edges = np.array(list(map(idMapidx.get, edges_undirect.flatten())),
                                dtype=np.int32).reshape(edges_undirect.shape)
            
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(len(idMapidx), len(idMapidx)),
                                dtype=np.float32)
            
            if no_attributes:
                nodes_num = len(idMapidx)
                if random_Attributes_dim:  
                    print('use random init feature!')
                    node_indices = np.random.permutation(np.arange(nodes_num))
                    features = sp.coo_matrix((np.ones(nodes_num), (np.arange(nodes_num), node_indices)), shape=(nodes_num, random_Attributes_dim))
                else:
                    print('use graph structure init feature!')
                    features = sp.eye(nodes_num)
                    features = adj + sp.eye(adj.shape[0])
            else:
                print('use origin feature!')
                pass
            features = self.normalize(features)
            features = torch.FloatTensor(np.array(features.todense()))
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj = self.normalize(adj + sp.eye(adj.shape[0]))
            Adj_dense = self.sparse_mx_to_torch_sparse_tensor(adj).to_dense()  
            adj = self.sparse_mx_to_torch_sparse_tensor(adj) 
            
            labels = torch.ones((1, 1))
            self.adj = adj
            self.features = features
            self.Adj_dense = Adj_dense
            
            self.save()
            
            return adj, features, None, Adj_dense
        
        elif self.dataset_name == 'ca-AstroPh':
            edges_path = self.dataset_path + '/' + 'CA-AstroPh.txt'
            feature_path = None
            nodes = []
            edges_undirect = []
            with open(edges_path, 'r') as fp:
                for edge in fp:
                    edge= edge.strip().split('\t')
                    edge = [int(i) for i in edge]
                    edges_undirect.append(edge)
                    nodes.extend(edge)
            nodes = list(set(nodes))
            
            idMapidx = {j:i for i,j in enumerate(nodes)}

            edges_undirect = np.array(edges_undirect)
            edges = np.array(list(map(idMapidx.get, edges_undirect.flatten())),
                                dtype=np.int32).reshape(edges_undirect.shape)
            
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(len(idMapidx), len(idMapidx)),
                                dtype=np.float32)
            
            if no_attributes:
                nodes_num = len(idMapidx)
                if random_Attributes_dim:  
                    print('use random init feature!')
                    node_indices = np.random.permutation(np.arange(nodes_num))
                    features = sp.coo_matrix((np.ones(nodes_num), (np.arange(nodes_num), node_indices)), shape=(nodes_num, random_Attributes_dim))
                else:
                    print('use graph structure init feature!')
                    features = sp.eye(nodes_num)
                    features = adj + sp.eye(adj.shape[0])
            else:
                print('use origin feature!')
                pass
            
            features = self.normalize(features)
            features = torch.FloatTensor(np.array(features.todense()))
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj = self.normalize(adj + sp.eye(adj.shape[0]))
            Adj_dense = self.sparse_mx_to_torch_sparse_tensor(adj).to_dense()  
            adj = self.sparse_mx_to_torch_sparse_tensor(adj) 
            
            labels = torch.ones((1, 1))
            
            self.adj = adj
            self.features = features
            self.Adj_dense = Adj_dense
            self.save()
            
            return adj, features, None, Adj_dense
        
        elif self.dataset_name == 'pubmed':
            num_nodes = 19717
            num_feats = 500
            feat_data = np.zeros((num_nodes, num_feats))
            labels = np.empty((num_nodes, 1), dtype=np.int64)
            node_map = {}
            with open(self.dataset_path + "/Pubmed-Diabetes.NODE.paper.tab") as fp:
                fp.readline()
                feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
                for i, line in enumerate(fp):
                    info = line.split("\t")
                    node_map[info[0]] = i
                    labels[i] = int(info[1].split("=")[1])-1
                    for word_info in info[2:-1]:
                        word_info = word_info.split("=")
                        feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
            adj_lists = []
            with open(self.dataset_path + "/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
                fp.readline()
                fp.readline()
                for line in fp:
                    info = line.strip().split("\t")
                    paper1 = info[1].split(":")[1]
                    paper2 = info[-1].split(":")[1]
                    adj_lists.append([paper1, paper2])
            idMapidx = node_map
            edges_undirect = np.array(adj_lists)
            edges = np.array(list(map(idMapidx.get, edges_undirect.flatten())),
                                dtype=np.int32).reshape(edges_undirect.shape)
            
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(len(idMapidx), len(idMapidx)),
                                dtype=np.float32)
            
            if no_attributes:
                nodes_num = len(idMapidx)
                if random_Attributes_dim:  
                    print('use random init feature!')
                    node_indices = np.random.permutation(np.arange(nodes_num))
                    features = sp.coo_matrix((np.ones(nodes_num), (np.arange(nodes_num), node_indices)), shape=(nodes_num, random_Attributes_dim))
                else:
                    print('use graph structure init feature!')
                    features = sp.eye(nodes_num)
                    features = adj + sp.eye(adj.shape[0])
                    
            else:
                print('use origin feature!')
                pass
            
            features = self.normalize(features)
            features = torch.FloatTensor(np.array(features.todense()))
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj = self.normalize(adj + sp.eye(adj.shape[0]))
            Adj_dense = self.sparse_mx_to_torch_sparse_tensor(adj).to_dense()  
            adj = self.sparse_mx_to_torch_sparse_tensor(adj) 
            
            labels = torch.ones((1, 1))
            
            self.adj = adj
            self.features = features
            self.Adj_dense = Adj_dense
            
            self.save()
            
            return adj, features, None, Adj_dense
    

        

    def save_features(self, features, file_path):
        torch.save(features, file_path)
    
    def load(self):
        print('[{}]Loading from exist[adj features labels Adj_dense]'.format(self.dataset_name))
        save_root_path = self.work_path+'/data/' + self.dataset_name + '/'
        self.adj = torch.load(save_root_path + 'adj/' + 'adj.pt')
        self.features = torch.load(save_root_path + 'features/' + 'features.pt')
        self.labels = torch.load(save_root_path + 'labels/' + 'labels.pt')
        self.Adj_dense = torch.load(save_root_path + 'Adj_dense/' + 'Adj_dense.pt')
    
    def save(self):
        # print('[{}]Saving...'.format(self.dataset_name))
        # save_root_path = self.work_path+'/data/' + self.dataset_name + '/'
        # torch.save(self.adj, save_root_path + 'adj/' + 'adj.pt')
        # torch.save(self.features, save_root_path + 'features/' + 'features.pt')
        # torch.save(self.labels, save_root_path + 'labels/' + 'labels.pt')
        # torch.save(self.Adj_dense, save_root_path + 'Adj_dense/' + 'Adj_dense.pt')
        pass

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """
        Convert a scipy sparse matrix to a torch sparse tensor.
        """
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)      
    
    def normalize(self, mx):
        """
        Row-normalize sparse matrix
        """
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    
    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                dtype=np.int32)
        return labels_onehot
    
    def target_distribution(q):
        weight = q**2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()
    



import torch
import networkx as nx
import numpy as np
import random
from tqdm import tqdm
import json
from copy import deepcopy
import collections


class Sampler:
    def __init__(self, args, path, dataset_name, kcore_num=3, sample_version=None): 
        self.dataset_root = path
        self.dataset_name = dataset_name
        self.dataset_path = path + '/' + dataset_name + '/'
        self.kcore_communities = None
        self.kcore_plus_communities = None
        self.sample_version = sample_version
        self.kcore_num = kcore_num
        self.mask_node_list = None
        self.work_path = args.work_path
        self.fake_sample_num = args.sampling_triplets_num
        self.degree_dict = None
        self.max_degree = None
    
    def get_nx_graph(self, graph_path, dataset, is_index_type=True):
        origin_edges, idx_map = self.load_graph()

        if is_index_type:
            edges = np.array(list(map(idx_map.get, origin_edges.flatten())), 
            dtype=np.int32).reshape(origin_edges.shape)
        else:
            edges = np.array(list(origin_edges), dtype=np.int32).reshape(origin_edges.shape)

        graph = nx.Graph()
        graph.add_edges_from(edges)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        return graph

    def get_degree_dict(self):
        graph = self.get_nx_graph(self.dataset_path, dataset=self.dataset_name)
        self.degree_dict = nx.degree(graph)
        unique_degrees = [v for k, v in self.degree_dict]
        self.max_degree = max(unique_degrees)
    
    def load_graph(self):
        if self.dataset_name == 'cora' or self.dataset_name == 'citeseer':
            idx_features_labels = np.genfromtxt("{}{}.content".format(self.dataset_path, self.dataset_name), dtype=np.dtype(str))
            edges_undirect = np.genfromtxt("{}{}.cites".format(self.dataset_path, self.dataset_name), dtype=np.dtype(str))
            id = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
            idMapidx = {j: i for i, j in enumerate(id)}
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
        
        elif self.dataset_name == 'ppi':
            feature_path = None
            edges_path = self.dataset_path + '/' + 'ppi-walks.txt'
            
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
        
        elif self.dataset_name == 'reddit':
            pass

    
        elif self.dataset_name == 'Amazon0302':
            # 读取边列表
            edges_path = self.dataset_path + '/' + 'amazon0302.txt'
            edges_undirect = []
            with open(edges_path, 'r') as fp:
                for line in fp:
                    if not line.startswith("#"):  # 忽略文件中的注释行
                        edge = line.strip().split('\t')
                        edge = [int(i) for i in edge]
                        edges_undirect.append(edge)
            edges_undirect = np.array(edges_undirect)
            nodes = np.unique(edges_undirect.flatten())
            idMapidx = {j: i for i, j in enumerate(nodes)}


        elif self.dataset_name == 'com-dblp':
            # 读取边列表
            edges_path = self.dataset_path + '/' + 'com-dblp.ungraph.txt'
            edges_undirect = []
            with open(edges_path, 'r') as fp:
                for line in fp:
                    if not line.startswith("#"):  # 忽略文件中的注释行
                        edge = line.strip().split('\t')
                        edge = [int(i) for i in edge]
                        edges_undirect.append(edge)
            edges_undirect = np.array(edges_undirect)
            nodes = np.unique(edges_undirect.flatten())
            idMapidx = {j: i for i, j in enumerate(nodes)}


        elif self.dataset_name == 'web-BerkStan':
            # 读取边列表
            edges_path = self.dataset_path + '/' + 'web-BerkStan.txt'
            edges_undirect = []
            with open(edges_path, 'r') as fp:
                for line in fp:
                    if not line.startswith("#"):  # 忽略文件中的注释行
                        edge = line.strip().split('\t')
                        edge = [int(i) for i in edge]
                        edges_undirect.append(edge)
            edges_undirect = np.array(edges_undirect)
            nodes = np.unique(edges_undirect.flatten())
            idMapidx = {j: i for i, j in enumerate(nodes)}

        
        return edges_undirect, idMapidx

    
    


        

    def get_different_kcore(self, need_outcore=True):
        k_value=self.kcore_num
        print('Get k-core [{}][k={}]'.format(self.dataset_name, k_value))

        try:
            graph = self.get_nx_graph(self.dataset_path, dataset=self.dataset_name)
            kcore = nx.k_core(G=graph, k=k_value)
            # ... 其余的方法代码 ...
        except Exception as e:
            print("Error in k_core computation:", e)
            if 'graph' in locals() or 'graph' in globals():
                print("Graph type:", type(graph))
                print("Graph info:", nx.info(graph))
            else:
                print("Graph variable is not defined.")
            print("K value:", k_value)
        
        graph = self.get_nx_graph(self.dataset_path, dataset=self.dataset_name)
        kcore = nx.k_core(G=graph, k=k_value)
        outside_kcore = set(graph.nodes()) - set(kcore.nodes())
        kcore_communities = {}
        for i, community in enumerate(nx.connected_components(kcore)):
            kcore_communities[i] = list(community)
        if need_outcore:
            kcore_communities[max(kcore_communities.keys()) + 1] = list(outside_kcore)

        self.kcore_communities = kcore_communities
        
    def get_different_kcore_plus(self, need_outcore=True):
        k_value=self.kcore_num + 1
        print('Get k-core [{}][k={}]'.format(self.dataset_name, k_value))
        
        graph = self.get_nx_graph(self.dataset_path, dataset=self.dataset_name)
        kcore = nx.k_core(G=graph, k=k_value)
        outside_kcore = set(graph.nodes()) - set(kcore.nodes())
        kcore_communities = {}
        for i, community in enumerate(nx.connected_components(kcore)):
            kcore_communities[i] = list(community)
        if need_outcore:
            kcore_communities[max(kcore_communities.keys()) + 1] = list(outside_kcore)

        self.kcore_plus_communities = kcore_communities
        
    def Query_Community(self, sample_num=1000, save=True, loadfromexist=True):
        if not loadfromexist:
            print('Random QC...')
            communities = self.kcore_communities
            communities_inner = []
            communities_outer = []
            for i in range(len(communities)):
                if i == (len(communities)-1):
                    communities_outer.extend(communities[i])
                else:
                    communities_inner.extend(communities[i])
            
            print("communities_inner: {}".format(len(communities_inner)))
            print("communities_outer: {}".format(len(communities_outer)))
            
            node_label = {}
            for i in range(len(communities)):
                for node in communities[i]:
                    node_label[node] = i
            
            communities_gt = {}
            community_type = communities.keys()
            
            for type in list(community_type):
                community_onehot = np.zeros([len(communities_inner)+len(communities_outer)])
                for i in communities[type]:
                    community_onehot[i] = 1
                communities_gt[type] = community_onehot

            Query_list = []
            Community_list = []
            nodes_idx = communities_inner
            for i in range(sample_num):
                query = nodes_idx.pop(random.randint(0, len(nodes_idx) - 1))
                Query_list.append(query)

            for i in Query_list:
                Community_list.append(communities_gt[node_label[i]])

            if save:
                np.save(self.work_path+'/data/'+ self.dataset_name +'/query_community/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']Query_list.npy', np.array(Query_list))
                # np.save(self.work_path+'/data/'+ self.dataset_name +'/query_community/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']Community_list.npy', np.array(Community_list))
                print("Save Successfully! [num: {}]".format(len(Query_list)))
            else:
                print("No Saving")
        else:
            print('[Random]Load from exist...')
            Query_list = np.load(self.work_path+'/data/'+ self.dataset_name +'/query_community/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']Query_list.npy')
            # Community_list = np.load(self.work_path+'/data/'+ self.dataset_name +'/query_community/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']Community_list.npy')
            Community_list = self.from_query_get_community(Query_list)
        return Query_list, Community_list
    
    def from_query_get_community(self, Query_list):
        communities = self.kcore_communities
        
        communities_inner = []
        communities_outer = []
        for i in range(len(communities)):
            if i == (len(communities)-1):
                communities_outer.extend(communities[i])
            else:
                communities_inner.extend(communities[i])
                
        communities_gt = {}
        community_type = communities.keys()
        
        node_label = {}
        for i in range(len(communities)):
            for node in communities[i]:
                node_label[node] = i
                    
        for type in list(community_type):
            community_onehot = np.zeros([len(communities_inner)+len(communities_outer)])
            for i in communities[type]:
                community_onehot[i] = 1
            communities_gt[type] = community_onehot
        
        Community_list = []
        for i in Query_list:
            Community_list.append(communities_gt[node_label[i]])
            
        return Community_list
    
    def Query_Community_all(self, save=True, loadfromexist=True):
        if not loadfromexist:
            print('All QC...')
            communities = self.kcore_communities
            communities_inner = []
            communities_outer = []
            for i in range(len(communities)):
                if i == (len(communities)-1):
                    communities_outer.extend(communities[i])
                else:
                    communities_inner.extend(communities[i])

            print("communities_inner: {}".format(len(communities_inner)))
            print("communities_outer: {}".format(len(communities_outer)))
            
            node_label = {}
            for i in range(len(communities)):
                for node in communities[i]:
                    node_label[node] = i
            
            communities_gt = {}
            community_type = communities.keys()
            
            for type in list(community_type):
                community_onehot = np.zeros([len(communities_inner)+len(communities_outer)])
                for i in communities[type]:
                    community_onehot[i] = 1
                communities_gt[type] = community_onehot

            all_Query_list = []
            all_Community_list = []
            all_Query_list = communities_inner

            for i in all_Query_list:
                all_Community_list.append(communities_gt[node_label[i]])

            if save:
                np.save(self.work_path+'/data/'+ self.dataset_name +'/query_community/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']all_Query_list.npy', np.array(all_Query_list))
                np.save(self.work_path+'/data/'+ self.dataset_name +'/query_community/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']all_Community_list.npy', np.array(all_Community_list))
                print("Save Successfully! [num: {}]".format(len(all_Query_list)))
            else:
                print("No Saving")
        else:
            print('[All]Load from exist...')
            all_Query_list = np.load(self.work_path+'/data/'+ self.dataset_name +'/query_community/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']all_Query_list.npy')
            all_Community_list = np.load(self.work_path+'/data/'+ self.dataset_name +'/query_community/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']all_Community_list.npy')
        
        return all_Query_list, all_Community_list

    def special_Query_Community(self, save=True, loadfromexist=True):
        if not loadfromexist:
            print('Special QC...')
            communities = self.kcore_communities
            communities_ = []
            for i in communities.keys():
                communities_.extend(communities[i])
                
            communities_inner = []
            communities_outer = []
            for i in range(len(communities)):
                if i == (len(communities)-1):
                    communities_outer.extend(communities[i])
                else:
                    if i == 0:
                        pass
                    else:
                        communities_inner.extend(communities[i])
            print("communities_inner: {}".format(len(communities_inner)))
            print("communities_outer: {}".format(len(communities_outer)))
            
            node_label = {}
            for i in range(len(communities)):
                for node in communities[i]:
                    node_label[node] = i
            
            communities_gt = {}
            community_type = communities.keys()
            
            for type in list(community_type):
                community_onehot = np.zeros([len(communities_)])
                for i in communities[type]:
                    community_onehot[i] = 1
                communities_gt[type] = community_onehot

            Query_list = []
            Community_list = []
            nodes_idx = communities_inner
            for i in range(len(communities_inner)):
                query = nodes_idx.pop(random.randint(0, len(nodes_idx) - 1))
                Query_list.append(query)

            for i in Query_list:
                Community_list.append(communities_gt[node_label[i]])

            if save:
                np.save(self.work_path+'/data/'+ self.dataset_name +'/query_community/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']special_Query_list.npy', np.array(Query_list))
                np.save(self.work_path+'/data/'+ self.dataset_name +'/query_community/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']special_Community_list.npy', np.array(Community_list))
                print("Save Successfully! [num: {}]".format(len(Query_list)))
            else:
                print("No Saving")
        else:
            print('[Special]Load from exist...')
            Query_list = np.load(self.work_path+'/data/'+ self.dataset_name +'/query_community/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']special_Query_list.npy')
            Community_list = np.load(self.work_path+'/data/'+ self.dataset_name +'/query_community/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']special_Community_list.npy')
        
        return Query_list, Community_list
    
    def get_triplet_datasets_degree(self, sample_num=3000, save=True, need_outcore=True, loadfromexist=True):
        if not loadfromexist:
            print('Triplets sampling...')
            communities = self.kcore_communities
            node_label = {}
            nodes_idx = []
            for i in range(len(communities)):
                for node in communities[i]:
                    node_label[node] = i
                    nodes_idx.append(node)
            community_type = list(communities.keys())
            print(community_type)
            
            length_list = [len(communities[i]) for i in community_type]
            avg_community_num = int(sum(length_list[:-1]) / (len(length_list)-1))
            
            self.get_degree_dict()
            for i in community_type:
                communities[i] = sorted(communities[i], key=lambda x: self.degree_dict[x], reverse=True)

            communities_better = deepcopy(communities)
            for i in community_type:
                communities_better[i] = communities_better[i][:int(length_list[i]*1.0)]

            anchor_list = []
            positive_list = []
            negative_list = []

            if len(community_type) == 2:
                for i in range(sample_num):
                    anchor = random.choice(communities[0])
                    neg_community_type = deepcopy(community_type)
                    neg_community_type.remove(node_label[anchor])
                    for i in neg_community_type:
                        anchor_list.append(anchor)
                        positive_list.append(random.choice(communities[node_label[anchor]]))
                        negative_list.append(random.choice(communities[i]))
            else:
                for i in range(sample_num):
                    anchor = random.choice(nodes_idx)
                    neg_community_type = deepcopy(community_type)
                    neg_community_type.remove(node_label[anchor])
                    for i in neg_community_type:
                        anchor_list.append(anchor)
                        positive_list.append(random.choice(communities_better[node_label[anchor]]))
                        negative_list.append(random.choice(communities_better[i]))

            print('Triplets num: {}'.format(len(anchor_list)))
            if save:
                if need_outcore:
                    np.save(self.work_path+'/triplets/'+ self.dataset_name +'/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']anchor_list.npy', anchor_list)
                    np.save(self.work_path+'/triplets/'+ self.dataset_name +'/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']positive_list.npy', positive_list)
                    np.save(self.work_path+'/triplets/'+ self.dataset_name +'/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']negative_list.npy', negative_list)
                    print("(Normal Way)(include outcore)Save Successfully!")
                else:
                    np.save(self.work_path+'/triplets/'+ self.dataset_name +'/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']anchor_list_no_outcore.npy', anchor_list)
                    np.save(self.work_path+'/triplets/'+ self.dataset_name +'/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']positive_list_no_outcore.npy', positive_list)
                    np.save(self.work_path+'/triplets/'+ self.dataset_name +'/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']negative_list_no_outcore.npy', negative_list)
                    print("(Normal Way)(no outcore)Save Successfully!")
        else:
            print('[Random]Load from exist...')
            anchor_list = np.load(self.work_path+'/triplets/'+ self.dataset_name +'/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']anchor_list.npy')
            positive_list = np.load(self.work_path+'/triplets/'+ self.dataset_name +'/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']positive_list.npy')
            negative_list = np.load(self.work_path+'/triplets/'+ self.dataset_name +'/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']negative_list.npy')

        return anchor_list, positive_list, negative_list


    def get_triplet_datasets(self, sample_num=3000, save=True, need_outcore=True, loadfromexist=True):
        if not loadfromexist:
            print('Triplets sampling...')
            communities = self.kcore_communities
            node_label = {}
            nodes_idx = []
            for i in range(len(communities)):
                for node in communities[i]:
                    node_label[node] = i
                    nodes_idx.append(node)
            community_type = list(communities.keys())
            print(community_type)

            anchor_list = []
            positive_list = []
            negative_list = []

            if len(community_type) == 2:
                for i in range(sample_num):
                    anchor = random.choice(communities[0])
                    neg_community_type = deepcopy(community_type)
                    neg_community_type.remove(node_label[anchor])
                    for i in neg_community_type:
                        anchor_list.append(anchor)
                        positive_list.append(random.choice(communities[node_label[anchor]]))
                        negative_list.append(random.choice(communities[i]))
            else:
                for i in range(sample_num):
                    anchor = random.choice(nodes_idx)
                    neg_community_type = deepcopy(community_type)
                    neg_community_type.remove(node_label[anchor])
                    for i in neg_community_type:
                        anchor_list.append(anchor)
                        positive_list.append(random.choice(communities[node_label[anchor]]))
                        negative_list.append(random.choice(communities[i]))
                        
            print('Triplets num: {}'.format(len(anchor_list)))
            if save:
                if need_outcore:
                    np.save(self.work_path+'/triplets/'+ self.dataset_name +'/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']anchor_list.npy', anchor_list)
                    np.save(self.work_path+'/triplets/'+ self.dataset_name +'/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']positive_list.npy', positive_list)
                    np.save(self.work_path+'/triplets/'+ self.dataset_name +'/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']negative_list.npy', negative_list)
                    print("(Normal Way)(include outcore)Save Successfully!")
                else:
                    np.save(self.work_path+'/triplets/'+ self.dataset_name +'/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']anchor_list_no_outcore.npy', anchor_list)
                    np.save(self.work_path+'/triplets/'+ self.dataset_name +'/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']positive_list_no_outcore.npy', positive_list)
                    np.save(self.work_path+'/triplets/'+ self.dataset_name +'/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']negative_list_no_outcore.npy', negative_list)
                    print("(Normal Way)(no outcore)Save Successfully!")
        else:
            print('[Random]Load from exist!')
            anchor_list = np.load(self.work_path+'/triplets/'+ self.dataset_name +'/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']anchor_list.npy')
            positive_list = np.load(self.work_path+'/triplets/'+ self.dataset_name +'/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']positive_list.npy')
            negative_list = np.load(self.work_path+'/triplets/'+ self.dataset_name +'/['+ str(self.sample_version) +'][k='+ str(self.kcore_num)+']negative_list.npy')

        return anchor_list, positive_list, negative_list

    
    def get_special_triplet_datasets_(self, sample_num=3000, save=True, need_outcore=True, loadfromexist=True):
        if not loadfromexist:
            communities = self.kcore_communities
            communities_plus = self.kcore_plus_communities
            
            community_type = list(communities.keys())
            communities_plus_type = list(communities_plus.keys())
            
            kcore_communities = deepcopy(communities)
            kcore_plus_communities = deepcopy(communities_plus)
            kcore_communities.popitem()
            kcore_plus_communities.popitem()
            
            # kcore_community_type = list(kcore_communities.keys())
            
            node_label = {}
            nodes_idx = []
            for key in communities.keys():
                for node in communities[key]:
                    node_label[node] = key
                    nodes_idx.append(node)
                    
            # communities_length_list = [len(v) for k,v in communities.items()]
            # communities_plus_length_list = [len(v) for k,v in communities_plus.items()]

            plus_node_label = {}
            plus_nodes_idx = []
            for key in communities_plus.keys():
                for node in communities_plus[key]:
                    plus_node_label[node] = key
                    plus_nodes_idx.append(node)

            # kcore_communities_length_list = [len(v) for k,v in communities.items()]
            kcore_plus_communities_length_list = [len(v) for k,v in communities_plus.items()]

            high_rank_kcore = kcore_plus_communities
            low_rank_kcore = kcore_communities
            all_low_rank_kcore_keys = low_rank_kcore.keys()
            old_key_set = set()
            special_key_set = set()
            new_key_set = set()
            
            old_key_dict = collections.defaultdict(list)

            for i in high_rank_kcore.keys():
                for j in low_rank_kcore.keys():
                    high_rank_kcore_set = set(high_rank_kcore[i])
                    low_rank_kcore_set = set(low_rank_kcore[j])
                    if high_rank_kcore_set < low_rank_kcore_set:
                        old_key_set.add(j)
                        old_key_dict[j].append(i)
                    elif high_rank_kcore_set == low_rank_kcore_set:
                        special_key_set.add(j)

            
            new_key_set = all_low_rank_kcore_keys - old_key_set - special_key_set

            old_key_dict_max = {}
            for key in old_key_dict.keys():
                old_key_dict_length = np.array([kcore_plus_communities_length_list[i] for i in old_key_dict[key]])
                old_key_dict_max[key] = old_key_dict[key][old_key_dict_length.argmax()]
                
            print(old_key_dict_max)
            
            origin_sampling_pool = communities
            anchor_sampling_pool = low_rank_kcore
            
            origin_pool_set = set()
            for i in low_rank_kcore.values():
                origin_pool_set = origin_pool_set | set(i)
            origin_pool_list = list(origin_pool_set)
            
            
            for key in old_key_dict_max.keys():
                difference_set = set(low_rank_kcore[key]) - set(high_rank_kcore[old_key_dict_max[key]])
                anchor_sampling_pool[key] = list(difference_set)
    
            anchor_pool_set = set()
            for i in anchor_sampling_pool.values():
                anchor_pool_set = anchor_pool_set | set(i)
            # anchor_pool_list = list(anchor_pool_set)
            
            old_sampling_pool = set()
            for i in old_key_dict_max.keys():
                old_sampling_pool = old_sampling_pool | set(high_rank_kcore[old_key_dict_max[key]])
            
            if len(new_key_set) == 0:
                pass
            else:
                for key in new_key_set:
                    temp_set = set(low_rank_kcore[key])
                    old_sampling_pool = old_sampling_pool | temp_set
            
            old_sampling_pool = list(old_sampling_pool)
            
            anchor_list = []
            positive_list = []
            negative_list = []
                     

            if len(community_type) == 2 and len(community_type) != len(communities_plus_type):
                sample_num = sample_num + self.fake_sample_num
                for i in range(sample_num):
                    anchor = random.choice(origin_pool_list)
                    neg_community_type = deepcopy(community_type)
                    neg_community_type.remove(node_label[anchor])
                    for i in neg_community_type:
                        anchor_list.append(anchor)
                        positive_list.append(random.choice(origin_sampling_pool[node_label[anchor]]))
                        negative_list.append(random.choice(origin_sampling_pool[i]))
            else:
                for i in range(sample_num):
                    anchor = random.choice(origin_pool_list)
                    neg_community_type = deepcopy(community_type)
                    neg_community_type.remove(node_label[anchor])
                    for i in neg_community_type:
                        anchor_list.append(anchor)
                        positive_list.append(random.choice(origin_sampling_pool[node_label[anchor]]))
                        negative_list.append(random.choice(origin_sampling_pool[i]))
            
            if len(community_type) == 2 and len(community_type) != len(communities_plus_type):
                pass
            else:
                for i in range(self.fake_sample_num):
                    anchor = random.choice(origin_sampling_pool[max(origin_sampling_pool.keys())])
                    neg_community_type = deepcopy(community_type)
                    neg_community_type.remove(node_label[anchor])
                    for i in neg_community_type:
                        anchor_list.append(anchor)
                        positive_list.append(random.choice(origin_sampling_pool[max(origin_sampling_pool.keys())]))
                        negative_list.append(random.choice(anchor_sampling_pool[i]))
                    
            
            
            self.mask_node_list = origin_sampling_pool[max(origin_sampling_pool.keys())]
            if save:
                np.save(self.work_path+'/triplets/'+ self.dataset_name +'/['+ str(self.sample_version) +'][k='+ str(self.kcore_num+1) + '->' + str(self.kcore_num) +']anchor_list.npy', anchor_list)
                np.save(self.work_path+'/triplets/'+ self.dataset_name +'/['+ str(self.sample_version) +'][k='+ str(self.kcore_num+1) + '->' + str(self.kcore_num) +']positive_list.npy', positive_list)
                np.save(self.work_path+'/triplets/'+ self.dataset_name +'/['+ str(self.sample_version) +'][k='+ str(self.kcore_num+1) + '->' + str(self.kcore_num) +']negative_list.npy', negative_list)
                print("({})Save Successfully Kcore[{}->{}]!".format(self.dataset_name, self.kcore_num+1, self.kcore_num))
    
        else:
            print('[Random]Load from exist...')
            anchor_list = np.load(self.work_path+'/triplets/'+ self.dataset_name +'/['+ str(self.sample_version) +'][k='+ str(self.kcore_num+1) + '->' + str(self.kcore_num) +']anchor_list.npy')
            positive_list = np.load(self.work_path+'/triplets/'+ self.dataset_name +'/['+ str(self.sample_version) +'][k='+ str(self.kcore_num+1) + '->' + str(self.kcore_num) +']positive_list.npy')
            negative_list = np.load(self.work_path+'/triplets/'+ self.dataset_name +'/['+ str(self.sample_version) +'][k='+ str(self.kcore_num+1) + '->' + str(self.kcore_num) +']negative_list.npy')

        return anchor_list, positive_list, negative_list
    
    
    def target_distribution(self, q):
        weight = q**2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()


    def get_community_vectors(self, communities:dict, all_embeddings):
        communnity_vectors = {}
        for community_key in communities.keys():
            communnity_vectors[community_key] = np.sum(all_embeddings[communities[community_key]], axis=0)/len(communities[community_key])
        return communnity_vectors
    
    def get_absolute_P(self):
        communities = self.kcore_communities
        nodes_idx = []
        for i in range(len(communities)):
            for node in communities[i]:
                nodes_idx.append(node)
        
        init_P = []
        for key in communities.keys():
            core_onehot = np.zeros([len(nodes_idx)])
            for i in communities[key]:
                core_onehot[i] = 1
            init_P.append(core_onehot)
        P = np.array(init_P).transpose()

class triplet_datasets():
    def __init__(self, anchor_list, positive_list, negative_list):
        self.anchor_list = torch.as_tensor(anchor_list)
        self.positive_list = torch.as_tensor(positive_list)
        self.negative_list = torch.as_tensor(negative_list)

    def __len__(self):
        return self.anchor_list.shape[0]

    def __getitem__(self, item):
        anchor = self.anchor_list[item]
        positive = self.positive_list[item]
        negative = self.negative_list[item]
        return anchor, positive, negative

    @staticmethod
    def collate_fn(batch):
        anchor, positive, negative = tuple(zip(*batch))
        anchor = torch.stack(anchor, dim=0)
        positive = torch.stack(positive, dim=0)
        negative = torch.stack(negative, dim=0)
        return anchor, positive, negative


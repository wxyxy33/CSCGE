import numpy as np
from csgcn.models.GCN_C import GCN_C

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from csgcn.analyze.validation_and_test_kcore import validation
from learning_schedule import adjust_learning_rate, adjust_learning_rate_

import time
import os
from tqdm import tqdm

from csgcn.analyze.validation_and_test_kcore import validation
from csgcn.data_process.data_processing import Dataset
from csgcn.data_process.data_sampling import Sampler, triplet_datasets

def train(args, experiment):
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    dataset = Dataset(args, args.work_path+'/data', args.dataset)
    adj, features, labels, Adj_dense = dataset.preprogress(no_attributes=True, random_Attributes_dim=None)

    # dataset.load()
    # adj = dataset.adj
    # features = dataset.features
    # labels = dataset.labels
    # Adj_dense  = dataset.Adj_dense
    
    sampler = Sampler(args, args.work_path+'/data', args.dataset, kcore_num=args.k_num, sample_version=args.sample_version)
    sampler.get_different_kcore(need_outcore=True)
    query_list, community_list = sampler.Query_Community(sample_num=args.query_community_num, save=True, loadfromexist=bool(args.loadfromexist_qc))
    
    print('load embedding from [{}]'.format(args.work_path+'/embedding/'+ args.dataset +'/pretraining_kcore/'+ str(experiment) +'/'+ '('+ str(args.round_num)+')'+'embedding_core' + str(args.k_num) + '.npy'))
    all_embeddings = np.load(args.work_path+'/embedding/'+ args.dataset +'/pretraining_kcore/'+ str(experiment) +'/'+ '('+ str(args.round_num)+')'+'embedding_core' + str(args.k_num) + '.npy')
    communities = sampler.kcore_communities
    args.clusters_num = len(communities)
    community_centra_vectors = sampler.get_community_vectors(communities, all_embeddings)
    communities_vector_array = np.array([community_centra_vectors[i] for i in range(len(community_centra_vectors))])
 
    model = GCN_C(nfeat=features.shape[1],
                    nhid=args.hidden,
                    # nclass=labels.max().item() + 1,
                    outdim=128,
                    dropout=args.dropout,
                    args=args)
    model.cluster_layer.data = F.normalize(torch.tensor(communities_vector_array)).cuda()
    # model.cluster_layer.data = torch.tensor(communities_vector_array).cuda()

    
    if args.k_plus_finetuning:
        print('Get kcore-plus community!')
        sampler.get_different_kcore_plus(need_outcore=True)

    optimizer = optim.Adam(model.parameters(), lr=0.1, eps=1e-3)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        # labels = labels.cuda()
        Adj_dense = Adj_dense.cuda()

    best_F1 = 0
    if args.k_plus_finetuning:
        print('Use special triplets')
        anchor_list, positive_list, negative_list = sampler.get_special_triplet_datasets_(sample_num=args.sampling_triplets_num, save=False, need_outcore=True, loadfromexist=bool(args.loadfromexist_sp))
    else:
        anchor_list, positive_list, negative_list = sampler.get_triplet_datasets(sample_num=args.sampling_triplets_num, save=False, need_outcore=True, loadfromexist=bool(args.loadfromexist_sp))

    triplet_data = triplet_datasets(anchor_list, positive_list, negative_list)
    print("Triplets' number: {}".format(len(anchor_list)))

    train_data_loader = torch.utils.data.DataLoader(triplet_data,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            collate_fn=triplet_data.collate_fn)

    embedding = []
    t_total = time.time()
    print("Training info:\nepoch[{}]\nbatch_size[{}]\nlearning_rate[{}]".format(args.epochs, args.batch_size, args.lr))
    print("Start finetuning!")
    
    max_epoch = args.epochs * len(train_data_loader)
    lr_min = 0.0000001
    lr_max = 0.01
    count = 0
    patience = 20
    delta = 0
    early_stop = False
    
    
    # start train
    for i in range(args.epochs):
        if early_stop:
            print(f"Early stop with [Epoch:{i}]")
            break
        data_loader = tqdm(train_data_loader, position=0, unit='B', unit_scale=True)
        for step, data in enumerate(data_loader):
            
            current_step = len(train_data_loader) * i + step
            adjust_learning_rate_(optimizer=optimizer, 
                                 current_epoch=current_step,
                                 max_epoch=max_epoch, 
                                 lr_min=lr_min, 
                                 lr_max=lr_max,
                                 warmup=True,
                                 i=i)
            
            if step == 0:
                model.eval()
                with torch.no_grad():
                    tmp_Q, _ = model(features, adj)
                    tmp_Q = tmp_Q.detach().data
                    # np.save(args.work_path+'/data/'+ args.dataset+'/QP_matrix/('+ str(args.round_num) +')Q_kcore('+ str(args.sample_version)+').npy', tmp_Q.cpu().numpy())
                    P = sampler.target_distribution(tmp_Q)
                    # np.save(args.work_path+'/data/'+ args.dataset+'/QP_matrix/('+ str(args.round_num) +')P_kcore('+ str(args.sample_version)+').npy', P.cpu().numpy())
            
            anchors_batch, positives_batch, negatives_batch = data

            model.train()
            optimizer.zero_grad()
            
            Q, x = model(features, adj)

            # get embedding
            anchors_embedding_batch = torch.stack([x[i] for i in anchors_batch], 0)
            positives_embedding_batch = torch.stack([x[i] for i in positives_batch], 0)
            negatives_batch_embedding_batch = torch.stack([x[i] for i in negatives_batch], 0)

            # Loss
            if args.distance == 'L2':
                triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(2), margin=1, reduction='mean')
            elif args.distance == 'Cosine':
                triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=1, reduction='mean')
            triplet_loss = triplet_loss(anchors_embedding_batch, positives_embedding_batch, negatives_batch_embedding_batch)
            clustering_loss = F.kl_div(Q.float().log(), P, reduction='batchmean')
            loss_train = triplet_loss + clustering_loss

            data_loader.set_description("Epoch[{}/{}] Batch[{}]".format(i+1, args.epochs, step+1))
            data_loader.set_postfix({'Loss' : '{0:1.5f}'.format(loss_train.item())})
            loss_train.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
            optimizer.step()

            x_ = x.detach().cpu().numpy()
            F1 = validation(x_, threshold=0.9, query_list=query_list, core_list=community_list)
            
            if F1 > best_F1:
                count = 0
                best_F1 = F1
                embedding = x_
                torch.save(model.state_dict(), args.work_path+'/model_weights/'+ args.dataset +'/triplet-pretrain-GCN(k='+ str(args.k_num) +')('+ str(args.round_num) +").pth")

                embedding_array = np.array(embedding)
                
                if not os.path.exists(args.work_path+'/embedding/'+ args.dataset +'/finetuning_TC/'+str(experiment)):
                    os.mkdir(args.work_path+'/embedding/'+ args.dataset +'/finetuning_TC/'+str(experiment))
                np.save(args.work_path+'/embedding/'+ args.dataset +'/finetuning_TC/'+ str(experiment) +'/'+ '('+ str(args.round_num)+')'+'embedding_core' + str(args.k_num) + '.npy', embedding_array)
            elif F1 < best_F1 + delta and i>2:
                count += 1
                if count >= patience:
                    early_stop =True
                    break    

    args.training_time = time.time() - t_total
    
    res = "[{}][{}][k={}] Time Consumption = {}s\n".format(args.Script_mode, args.dataset, args.k_num, args.training_time)
    file = "../Result/"+"[training_time]Res"
    with open(file, 'a+') as f:
        f.write(res)
    
    print("Optimization Finished!")
    print("Experiment {} done!".format(experiment))
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print("\n")
    return embedding


    



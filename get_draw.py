import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import os
import time

def Cal_F1(C_q_list, Y_q_list):
    
    sum_C = torch.sum(torch.sum(C_q_list, 1))
    sum_Y = torch.sum(torch.sum(Y_q_list, 1))
    C_Y = C_q_list + Y_q_list
    C_Y = (C_Y == 2).float()
    sum_C_Y = torch.sum(torch.sum(C_Y, 1))
    
    rec = sum_C_Y / sum_C
    pre = sum_C_Y / sum_Y
    F1 = (2 * pre * rec ) / (pre + rec) 

    return F1, pre, rec

k_values = []
f1_gcn_values, precision_gcn_values, recall_gcn_values = [], [], []
f1_decode_values, precision_decode_values, recall_decode_values = [], [], []


if __name__ == "__main__":

    current_path = os.getcwd()
    total_query_time = 0
    total_queries = 0
    query_times = []

    dataset = "cora"
    # fine_tune = "pretraining_kcore"
    fine_tune = "finetuning_TC"
    Q_C_path = "./offline/"+dataset+"/query_community/"
    emb_path = "./offline/embedding/"+dataset+"/"+fine_tune+"/0/"
    k_ini = 2
    k_bgn = k_ini
    k_end = 4
    k_max = 4
    k_total = k_end - k_ini+2


    for mode in ['ORI', 'REC']:
        file = open(str(current_path) + "/" + dataset +"_draw.txt", "a")
        file.close()
        k_bgn = k_ini

        for i in range(1, k_total):

            query_list_path = np.load(Q_C_path + "[0][k=" + str(k_bgn) + "]Query_list.npy")
            core_list_path = np.load(Q_C_path + "[k=" + str(k_bgn) + "]community_list.npy")

            if(mode == 'ORI'):
                embedding = np.load(emb_path + "("+str(k_max)+")embedding_core" + str(k_bgn) + ".npy")

            elif(mode == 'REC'):
                embedding = np.load(current_path + "/save_emb/decoded_embeddings_100_label_"+str(k_bgn) +".npy")
            
            embedding = torch.Tensor(embedding)
            threshold = 0.9
            core_list_onehot = []
            for core in core_list_path:
                core_onehot = torch.Tensor(core)
                core_list_onehot.append(core_onehot)

            core_list_onehot = torch.stack([i for i in core_list_onehot], 0)
    
            query_community_onehot = []
            start_time = time.time()  # Start time before processing queries
            num_queries = len(query_list_path)  # Number of queries to be processed
            for query in query_list_path:
                query_matrix = embedding[query].expand(embedding.shape[0], 128)
                cosine_similarity = F.cosine_similarity(query_matrix, embedding)
                Q_community = (cosine_similarity>threshold).float()
                query_community_onehot.append(Q_community)

            query_community_onehot = torch.stack([i for i in query_community_onehot], 0)
        
            end_time = time.time()  # End time after processing queries
            total_time = end_time - start_time  # Total time for processing all queries at this k-core
            average_time_per_query = total_time / num_queries if num_queries > 0 else 0
            query_times.append(average_time_per_query)  # Append average time to the list
            
            with open(f"{current_path}/{dataset}_query_time.txt", "a") as file:
                file.write(f"k-core: {k_bgn}, Average Query Time: {average_time_per_query:.4f} seconds\n")
            
            total_query_time += total_time  # Accumulate total time for all queries
            total_queries += num_queries  # Accumulate total number of queries processed
    
            F1, pre, rec = Cal_F1(core_list_onehot, query_community_onehot)
            print("Mode:", mode, "k-core:", k_bgn, "F1:", F1, "precision: ", pre, "recall: ", rec)

            if mode == 'ORI':
                k_values.append(k_bgn)
                f1_gcn_values.append(F1.item())
                precision_gcn_values.append(pre.item())
                recall_gcn_values.append(rec.item())
            elif mode == 'REC':
                f1_decode_values.append(F1.item())
                precision_decode_values.append(pre.item())
                recall_decode_values.append(rec.item())
            file = open(str(current_path) + "/" + dataset +"_draw.txt", "a")
            file.write("\nk-core num = " + str(k_bgn) + " threshold: " + str(threshold))
            file.write("\nF1: " + str(F1))
            file.write("\nRecall: " + str(rec))
            file.write("\nPresicion: " + str(pre))
            file.write("\n")
            file.close()

            k_bgn = k_bgn + 1
            
            if k_bgn == k_max:
                overall_average_query_time = total_query_time / total_queries if total_queries > 0 else 0

                # Write the overall average time to the same file
                with open(f"{current_path}/{dataset}_query_time.txt", "a") as file:
                    file.write(f"Overall Average Query Time: {overall_average_query_time:.8f} seconds\n")

                # Print the overall average query time for immediate visibility
                print(f"Overall Average Query Time: {overall_average_query_time:.8f} seconds")

    average_f1_gcn = sum(f1_gcn_values) / len(f1_gcn_values)
    average_f1_decode = sum(f1_decode_values) / len(f1_decode_values)

    # Print average F1 scores
    print(f"Average F1 score for 'GCN': {average_f1_gcn:.4f}")
    print(f"Average F1 score for 'decode': {average_f1_decode:.4f}")

    file = open(str(current_path) + "/" + dataset +"_draw.txt", "a")
    file.write("\naverage_f1_gcn = " + str(average_f1_gcn))
    file.write("\naverage_f1_decode = " + str(average_f1_decode))
    # file.write("\nF1: " + str(F1))
    file.write("\n")
    file.close()

    plt.figure(1)
    plt.ylim(0.6, 1.0)
    plt.plot(k_values, f1_gcn_values, label='version 1: GE+FT+KL', color='blue', marker='o')
    plt.plot(k_values, f1_decode_values, label='version 2: GE+FT+KL+ED', color='red', marker='x')
    plt.legend()
    plt.title('F1 Score Comparison')
    plt.xlabel('k-core number')
    plt.ylabel('F1 Score')
    plt.savefig('./img/f1_score_comparison.png')


    plt.figure(2)
    plt.ylim(0.6, 1.0)
    plt.plot(k_values, precision_gcn_values, label='version 1: GE+FT+KL', color='green', marker='o')
    plt.plot(k_values, precision_decode_values, label='version 2: GE+FT+KL+ED', color='purple', marker='x')
    plt.legend()
    plt.title('Precision Comparison')
    plt.xlabel('k-core number')
    plt.ylabel('Precision')
    plt.savefig('./img/precision_comparison.png')


    plt.figure(3)
    plt.ylim(0.6, 1.0)
    plt.plot(k_values, recall_gcn_values, label='version 1: GE+FT+KL', color='orange', marker='o')
    plt.plot(k_values, recall_decode_values, label='version 2: GE+FT+KL+ED', color='pink', marker='x')
    plt.legend()
    plt.title('Recall Comparison')
    plt.xlabel('k-core number')
    plt.ylabel('Recall')
    plt.savefig('./img/recall_comparison.png')

    plt.show()


    plt.figure(1)
    plt.ylim(0.6, 1.0)
    plt.plot(k_values, f1_gcn_values, label='version 1: GE+FT+KL', color='blue', marker='o')
    plt.plot(k_values, f1_decode_values, label='version 2: GE+FT+KL+ED', color='red', marker='x')


    for i, (x, y) in enumerate(zip(k_values, f1_gcn_values)):
        plt.text(x, y, f'{y:.2f}', color='blue', ha='center', va='bottom')
    for i, (x, y) in enumerate(zip(k_values, f1_decode_values)):
        plt.text(x, y, f'{y:.2f}', color='red', ha='center', va='bottom')

    plt.legend()
    plt.title('F1 Score Comparison')
    plt.xlabel('k-core number')
    plt.ylabel('F1 Score')
    plt.savefig('./img_value/f1_score_comparison.png')


    plt.figure(2)
    plt.ylim(0.6, 1.0)
    plt.plot(k_values, precision_gcn_values, label='version 1: GE+FT+KL', color='green', marker='o')
    plt.plot(k_values, precision_decode_values, label='version 2: GE+FT+KL+ED', color='purple', marker='x')


    for i, (x, y) in enumerate(zip(k_values, precision_gcn_values)):
        plt.text(x, y, f'{y:.2f}', color='green', ha='center', va='bottom')
    for i, (x, y) in enumerate(zip(k_values, precision_decode_values)):
        plt.text(x, y, f'{y:.2f}', color='purple', ha='center', va='bottom')

    plt.legend()
    plt.title('Precision Comparison')
    plt.xlabel('k-core number')
    plt.ylabel('Precision')
    plt.savefig('./img_value/precision_comparison.png')


    plt.figure(3)
    plt.ylim(0.6, 1.0)
    plt.plot(k_values, recall_gcn_values, label='version 1: GE+FT+KL', color='orange', marker='o')
    plt.plot(k_values, recall_decode_values, label='version 2: GE+FT+KL+ED', color='pink', marker='x')


    for i, (x, y) in enumerate(zip(k_values, recall_gcn_values)):
        plt.text(x, y, f'{y:.2f}', color='orange', ha='center', va='bottom')
    for i, (x, y) in enumerate(zip(k_values, recall_decode_values)):
        plt.text(x, y, f'{y:.2f}', color='pink', ha='center', va='bottom')

    plt.legend()
    plt.title('Recall Comparison')
    plt.xlabel('k-core number')
    plt.ylabel('Recall')
    plt.savefig('./img_value/recall_comparison.png')

    plt.show()

    

dataset="citeseer"
Script_mode="pretraining_kcore"
sample_version=0

loadfromexist_qc=0
loadfromexist_sp=0
index=(0 1 2 3 4)
# index=(2 3 4)
k_num=(6 5 4 3 2)
round_num=70

query_community_num=(20 50 150 500 1000)
sampling_triplets_num=(20000 10000 8000 8000 8000)

device=0


# python main.py\
#             --flag=1\
#             --Script_mode=${Script_mode}\
#             --distance=Cosine\
#             --epochs=20\
#             --batch_size=2048\
#             --k_num=${round_num}\
#             --round_num=${round_num}\
#             --sample_version=${sample_version}\
#             --loadfromexist_qc=1\
#             --loadfromexist_sp=0\
#             --dataset=${dataset}\
#             --k_plus_finetuning=0\
#             --query_community_num=10\
#             --sampling_triplets_num=30000\
#             --device=${device}

# python main.py\
#     --flag=0\
#     --Script_mode=${Script_mode}\
#     --k_num=${round_num}\
#     --round_num=${round_num}\
#     --sample_version=${sample_version}\
#     --dataset=${dataset}\
#     --sampling_triplets_num=30000\
#     --k_plus_finetuning=0

for i in ${index[*]}
    do
        python main.py\
            --flag=1\
            --Script_mode=${Script_mode}\
            --distance=Cosine\
            --epochs=20\
            --batch_size=2048\
            --k_num=${k_num[${i}]}\
            --round_num=${round_num}\
            --sample_version=${sample_version}\
            --loadfromexist_qc=0\
            --loadfromexist_sp=0\
            --dataset=${dataset}\
            --k_plus_finetuning=0\
            --query_community_num=${query_community_num[${i}]}\
            --sampling_triplets_num=${sampling_triplets_num[${i}]}\
            --device=${device}
    done

for i in ${index[*]}
    do
        python main.py\
            --flag=0\
            --Script_mode=${Script_mode}\
            --k_num=${k_num[${i}]}\
            --round_num=${round_num}\
            --sample_version=${sample_version}\
            --dataset=${dataset}\
            --k_plus_finetuning=0\
            --query_community_num=${query_community_num[${i}]}\
            --sampling_triplets_num=${sampling_triplets_num[${i}]}
    done

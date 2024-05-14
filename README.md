### Prepare for experiments

Simply run `conda create --name myenv --file requirements.txt`.

### Generate graph embeddings

1. You can start with run.sh, and change the mode for finetuning, KL optimization and dataset for desired k-level embedding.

2. Or you can run the code and get one desired graph embedding.
   Eg. Here we get the graph embedding for 4-core community search for citeseer dataset(the maximum k-core is 7.)

   `python main.py\
               --Script_mode=pretraining_kcore\
               --distance=Cosine\
               --epochs=20\
               --batch_size=2048\
               --k_num=4\
               --round_num=7\
               --sample_version=0\
               --loadfromexist_qc=0\
               --loadfromexist_sp=0\
               --dataset=citeseer\
               --k_plus_finetuning=0\
               --query_community_num=20\
               --sampling_triplets_num=8000\
               --device=1`



### Compress graph embeddings

1. Run the code and get the compressed graph embeddings and Encoder-Decoder weights.

   `python ED_model.py`

2. Run the code and test the quality of compressed graph embeddings in the community search stage.

   `python get_draw.py`

   

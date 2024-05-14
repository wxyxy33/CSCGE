### Setting Up the Environment

1. **Create a Python Environment:**
   - Use the following command to create a Conda environment using the `requirements.txt` file. This file should contain all the necessary packages.
     `conda create --name myenv --file requirements.txt`
   - Activate the environment:
     `conda activate myenv`



### Generate Graph Embeddings

1. **Start with `run.sh`:**

- You can initiate the process using `run.sh` and modify the script to set the mode for finetuning, KL optimization, and select the dataset for generating graph embeddings as required.

2. **Run the Code for Specific Graph Embedding:**

- Example command to generate graph embeddings for the 4-core community search on the Citeseer dataset:
  `python main.py \
              --Script_mode=pretraining_kcore \
              --distance=Cosine \
              --epochs=20 \
              --batch_size=2048 \
              --k_num=4 \
              --round_num=7 \
              --sample_version=0 \
              --loadfromexist_qc=0 \
              --loadfromexist_sp=0 \
              --dataset=citeseer \
              --k_plus_finetuning=0 \
              --query_community_num=20 \
              --sampling_triplets_num=8000 \
              --device=1`

### Compress Graph Embeddings

1. **Generate Compressed Embeddings and Model Weights:**
   `python ED_model.py`
2. **Evaluate Compressed Embeddings:**
   - Test the quality of the compressed graph embeddings during the community search stage:
     `python get_draw.py`




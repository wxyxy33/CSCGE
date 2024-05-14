import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 设定随机种子
seed = 33
torch.manual_seed(seed)

# 数据加载
emb_path = "./offline/embedding/"
dataset = "cora"
version = 4
k_ini = 2
k_end = 4
k_max = 4
k_num = k_end - k_ini + 1
if version == 1:
    train_mode = "pretraining_kcore"
    emb_name = "(" + str(k_max) + "0" + ")"
elif version == 2:
    train_mode = "pretraining_kcore"
    emb_name = "(" + str(k_max) + ")"
elif version == 3:
    train_mode = "finetuning_TC"
    emb_name = "(" + str(k_max) + "0" + ")"
elif version == 4:
    train_mode = "finetuning_TC"
    emb_name = "(" + str(k_max) + ")"



# 加载嵌入
files_names = [emb_path + dataset + "/" + train_mode +"/0/"+ emb_name + "embedding_core" + f"{i}" + ".npy" for i in range(k_ini, k_end+1)]
embeddings_list = [np.load(files_name) for files_name in files_names]
embeddings = np.stack(embeddings_list, axis=2)  # (18772, 128, 7)
y_size = embeddings.shape[0]
# print(y_size)
embeddings = torch.tensor(embeddings, dtype=torch.float32)
embeddings = embeddings.to(device)
dataset = TensorDataset(embeddings)

# 定义批量大小
batch_size = 1024 

# 创建 DataLoader
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_weights = torch.softmax(Q @ K.transpose(-2, -1) / (K.size(-1) ** 0.5), dim=-1)
        attention_output = attention_weights @ V
        return self.norm(attention_output + x)


    
class ConditionalAttention(nn.Module):
    def __init__(self, embedding_dim, k_num):
        super(ConditionalAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.k_num = k_num
        self.fc1 = nn.Linear(embedding_dim + k_num, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, fused_embeddings, labels):
        # 将标签信息和编码后的嵌入合并
        combined = torch.cat((fused_embeddings, labels), dim=1)
        # 通过一个全连接层
        x = F.relu(self.fc1(combined))
        # 计算注意力权重
        attention_weights = torch.softmax(self.fc2(x), dim=1)
        # 应用注意力权重
        weighted_embeddings = attention_weights * fused_embeddings
        return weighted_embeddings

class CNNEncoder(nn.Module):
    def __init__(self, embedding_dim, k_num, cnn_output_dim):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=cnn_output_dim, kernel_size=1)

    def forward(self, x):
        # x shape: (batch_size, embedding_dim, k_num)
        x = x.permute(0, 2, 1)  # (batch_size, k_num, embedding_dim)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # print("CNNEnocder shape: "+str(x.shape))
        return x

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        # x shape: (batch_size, k_num, input_dim)
        # print("lstm input shape: "+str(x.shape))

        _, (hn, _) = self.lstm(x)  # hn 是每层的最后时间步的隐藏状态
        hn_last_layer = hn[-1, :, :]  # 取最后一层的隐藏状态

        return hn_last_layer

        # x, (hn, cn) = self.lstm(x)
        # return hn[-1]
    
    

# class LSTMEncoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0):
#         super(LSTMEncoder, self).__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)


#     def forward(self, x):
#         x = x.permute(0, 2, 1)  # 转换为(batch_size, seq_len, input_dim)
#         _, (hn, _) = self.lstm(x)  # hn 是每层的最后时间步的隐藏状态
#         hn_last_layer = hn[-1, :, :]  # 取最后一层的隐藏状态

#         return hn_last_layer

class CombinedEncoder(nn.Module):
    def __init__(self, embedding_dim, k_num, cnn_output_dim, lstm_hidden_dim):
        super(CombinedEncoder, self).__init__()
        self.cnn_encoder = CNNEncoder(embedding_dim, k_num, cnn_output_dim)
        self.lstm_encoder = LSTMEncoder(cnn_output_dim, lstm_hidden_dim)
        self.self_attention = SelfAttention(lstm_hidden_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, k_num)
        cnn_output = self.cnn_encoder(x)
        cnn_output = cnn_output.permute(0, 2, 1) 
        lstm_output = self.lstm_encoder(cnn_output)
        # print("CNN output shape:", cnn_output.shape)
        # print("LSTM input shape:", lstm_output.shape)

        attention_output = self.self_attention(lstm_output)
        return attention_output

# class DecoderLSTM(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim, k_num):
#         super(DecoderLSTM, self).__init__()
#         self.lstm = nn.LSTM(embedding_dim + k_num, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, embedding_dim)

#     def forward(self, encoded, labels):
#         x = torch.cat((encoded, labels), dim=1)
#         x, _ = self.lstm(x.unsqueeze(1))
#         x = self.fc(x.squeeze(1))
#         return x
    

class DecoderLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, k_num, num_layers=2):
        super(DecoderLSTM, self).__init__()
        self.lstm = nn.LSTM(embedding_dim + k_num, hidden_dim, num_layers, batch_first=True)
        self.conditional_attention = ConditionalAttention(embedding_dim, k_num)
        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, fused_embeddings, labels):
        x = torch.cat((fused_embeddings, labels), dim=1)
        x, (hidden, _) = self.lstm(x.unsqueeze(1))
        x = x.squeeze(1)
        # 应用条件注意力
        x = self.conditional_attention(x, labels)
        output_embeddings = self.fc(x)
        return output_embeddings

encoder = CombinedEncoder(embedding_dim=128, k_num=k_num, cnn_output_dim=64, lstm_hidden_dim=128).to(device)
decoder = DecoderLSTM(embedding_dim=128, hidden_dim=128, k_num=k_num).to(device)

# encoder.load_state_dict(torch.load('./save_model/encoder_model_7000epochs.pth'))
# decoder.load_state_dict(torch.load('./save_model/decoder_model_7000epochs.pth'))

criterion = nn.CosineEmbeddingLoss()
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0001)

num_epochs = 100  # 总的训练周期数

writer = SummaryWriter(log_dir=f"{os.getcwd()}/log/{num_epochs}epochs/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
print(" Now is model LSTM.py run for "+str(num_epochs)+" epochs.")




# 训练循环

start_time = datetime.now()
for epoch in tqdm(range(num_epochs), desc='Epochs', unit='epoch'):
    with tqdm(total=len(data_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        for batch in data_loader:
            embeddings_batch = batch[0].to(device)  # (batch_size, k_num, embedding_dim)

            optimizer.zero_grad()
            encoded_embeddings = encoder(embeddings_batch)

            total_loss = 0
            for i in range(k_num):
                label_vector = torch.zeros(embeddings_batch.size(0), k_num).to(device)
                label_vector[:, i] = 1
                decoded_embeddings = decoder(encoded_embeddings, label_vector)

                embeddings_batch_i = embeddings_batch[:, :, i]
                embeddings_batch_i = embeddings_batch_i.view(-1, 128)
                loss = criterion(decoded_embeddings, embeddings_batch_i, torch.ones(embeddings_batch.size(0), device=device))
                total_loss += loss

            avg_loss = total_loss / k_num
            writer.add_scalar('average_training_loss', avg_loss.item(), epoch)
            avg_loss.backward()
            optimizer.step()

            pbar.update(1)  # 更新内层进度条
            pbar.set_postfix({'Average Loss': avg_loss.item()})

    # 可选：在每个epoch结束后打印平均损失
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss.item()}')

end_time = datetime.now()
# # print("Training completed at", end_time.strftime("%Y-%m-%d %H:%M:%S"))
# total_training_time = end_time - start_time
# print(f"Total training time: {total_training_time}")
# with open(f"./train_time.txt", "a") as file:
#     file.write(f"Overall Average Query Time: {total_training_time:.8f} seconds\n")
    
total_training_time = end_time - start_time
total_seconds = total_training_time.total_seconds()
print(f"Total training time: {total_seconds:.8f} seconds")
with open(f"./train_time.txt", "a") as file:
    file.write(f"Total training time: {total_seconds:.8f} seconds")
# 保存模型
torch.save(encoder.state_dict(), './save_model/encoder_model_'+str(num_epochs)+'epochs.pth')
torch.save(decoder.state_dict(), './save_model/decoder_model_'+str(num_epochs)+'epochs.pth')


# 设置评估模式
encoder.eval()
decoder.eval()

# 假设 embeddings 的形状为 [batch_size, input_dim, seq_len]
# 为避免内存溢出，分批处理数据
batch_size_for_saving = 1024  # 可以根据您的 GPU 容量调整
num_batches = int(np.ceil(embeddings.size(0) / batch_size_for_saving))

# 用于收集所有批次的解码嵌入结果
fused_embeddings_all = np.zeros((embeddings.size(0), 128))
decoded_embeddings_all = [np.zeros((embeddings.size(0), 128)) for _ in range(k_num)]

for batch_idx in range(num_batches):
    # 计算当前批次的开始和结束索引
    start_idx = batch_idx * batch_size_for_saving
    end_idx = start_idx + batch_size_for_saving
    batch_embeddings = embeddings[start_idx:end_idx].to(device)

    with torch.no_grad():
        # 处理当前批次的数据
        fused_embeddings = encoder(batch_embeddings)  # 形状为 [batch_size, embedding_dim]
        fused_embeddings_all[start_idx:end_idx] = fused_embeddings.cpu().numpy()

        for i in range(k_num):
            # 为每个标签创建一个标签向量
            label_vector = torch.zeros(batch_embeddings.size(0), k_num).to(device)
            label_vector[:, i] = 1

            # 使用解码器得到针对特定标签的解码嵌入
            decoded_embeddings = decoder(fused_embeddings, label_vector)  # 形状应为 [batch_size, embedding_dim]

            # 将结果转移到 CPU 并累积
            decoded_embeddings_all[i][start_idx:end_idx] = decoded_embeddings.cpu().numpy()

# 所有批次处理完成后，保存最终结果
np.save(f'./save_emb/fusion_embeddings_{num_epochs}.npy', fused_embeddings_all)
for i in range(k_num):
    np.save(f'./save_emb/decoded_embeddings_{num_epochs}_label_{i+k_ini}.npy', decoded_embeddings_all[i])

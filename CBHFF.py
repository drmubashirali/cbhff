#Cascaded BiLSTM Hierarchical Feature Fusion
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import math

def MARD(pred_values,ref_values):
    pred_values = pred_values.reshape(-1)
    ref_values = ref_values.reshape(-1)
    data_length = len(ref_values)    
    total = 0
    for i in range(data_length):
        temp = abs((ref_values[i] - pred_values[i]) / ref_values[i])
        total = total + temp
    mard_value = total / data_length
    mard_value = mard_value*100
    print('MARD: ',mard_value)
        
    return mard_value

def RMSE(pred_values,ref_values):
    pred_values = pred_values.reshape(-1)
    ref_values = ref_values.reshape(-1)
    data_length = len(ref_values)
    total = 0
    for i in range (data_length):
        temp = (ref_values[i] - pred_values[i]) * (ref_values[i] - pred_values[i])
        total = total + temp

    smse_value = math.sqrt(total / data_length)
    print('RMSE: ',smse_value)
        
    return smse_value

def clarke_error_zone_detailed(act, pred):
    if (act < 70 and pred < 70) or abs(act - pred) < 0.2 * act:
        return 0
    # Zone E - left upper
    if act <= 70 and pred >= 180:
        return 8
    # Zone E - right lower
    if act >= 180 and pred <= 70:
        return 7
    # Zone D - right
    if act >= 240 and 70 <= pred <= 180:
        return 6
    # Zone D - left
    if act <= 70 <= pred <= 180:
        return 5
    # Zone C - upper
    if 70 <= act <= 290 and pred >= act + 110:
        return 4
    # Zone C - lower
    if 130 <= act <= 180 and pred <= (7/5) * act - 182:
        return 3
    # Zone B - upper
    if act < pred:
        return 2
    # Zone B - lower
    return 1


clarke_error_zone_detailed = np.vectorize(clarke_error_zone_detailed)

def zone_accuracy(act_arr, pred_arr, mode='clarke', detailed=False, diabetes_type=1):
    
    acc = np.zeros(9)
    if mode == 'clarke':
        res = clarke_error_zone_detailed(act_arr, pred_arr)
    else:
        raise Exception('Unsupported error grid mode')

    acc_bin = np.bincount(res)
    acc[:len(acc_bin)] = acc_bin

    if not detailed:
        acc[1] = acc[1] + acc[2]
        acc[2] = acc[3] + acc[4]
        acc[3] = acc[5] + acc[6]
        acc[4] = acc[7] + acc[8]
        acc = acc[:5]
    score = acc / sum(acc)
    print('CEG:  A:{:.4f}, B:{:.4f}, C:{:.4f}, D:{:.4f}, E:{:.4f}'.format(score[0], score[1], score[2], score[3],score[4]))
    return score
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module('flatten', Flatten())
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module('gate_c_fc_%d' % i, nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_bn_%d' % (i + 1), nn.BatchNorm1d(gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_relu_%d' % (i + 1), nn.ReLU())
        self.gate_c.add_module('gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, in_tensor):
        avg_pool = F.avg_pool1d(in_tensor, in_tensor.size(2), stride=in_tensor.size(2)).squeeze(-1)
        channel_attention = self.gate_c(avg_pool).unsqueeze(2)
        return channel_attention

class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module('gate_s_conv_reduce0',
                               nn.Conv1d(gate_channel, gate_channel // reduction_ratio, kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0', nn.BatchNorm1d(gate_channel // reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())
        for i in range(dilation_conv_num):
            self.gate_s.add_module('gate_s_conv_di_%d' % i,
                                   nn.Conv1d(gate_channel // reduction_ratio, gate_channel // reduction_ratio,
                                             kernel_size=3, padding=dilation_val, dilation=dilation_val))
            self.gate_s.add_module('gate_s_bn_di_%d' % i, nn.BatchNorm1d(gate_channel // reduction_ratio))
            self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())
        self.gate_s.add_module('gate_s_conv_final', nn.Conv1d(gate_channel // reduction_ratio, 1, kernel_size=1))

    def forward(self, in_tensor):
        spatial_attention = self.gate_s(in_tensor)
        spatial_attention = spatial_attention.squeeze(1).unsqueeze(1) 
        return spatial_attention

class Multiview_Attention(nn.Module):
    def __init__(self, gate_channel):
        super(Multiview_Attention, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)

    def forward(self, in_tensor):
        channel_attention = self.channel_att(in_tensor)
        spatial_attention = self.spatial_att(in_tensor)
       
        attention_map = channel_attention + spatial_attention
       
        attention_map = torch.sigmoid(attention_map)
       
        multiplied_output = in_tensor * attention_map
      
        output = in_tensor + multiplied_output
        return output

class BloodGlucosePredictor(nn.Module):
    def __init__(self):
        super(BloodGlucosePredictor, self).__init__()
       
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.mva1 = Multiview_Attention(gate_channel=16)  
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.mva2 = Multiview_Attention(gate_channel=32) 
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.mva3 = Multiview_Attention(gate_channel=64)  
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)
        self.dropout3 = nn.Dropout(0.25)

        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.mva4 = Multiview_Attention(gate_channel=128)  
        self.bn4 = nn.BatchNorm1d(128)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # BiLSTM Blocks
        self.lstm_a = nn.LSTM(input_size=128, hidden_size=4, batch_first=True, bidirectional=True)
        self.lstm_b = nn.LSTM(input_size=128, hidden_size=2, batch_first=True, bidirectional=True)
        self.lstm_c = nn.LSTM(input_size=128, hidden_size=1, batch_first=True, bidirectional=True)

        common_size = 128
        self.transform_a = nn.Linear(4 * 2, common_size)
        self.transform_b = nn.Linear(2 * 2, common_size)
        self.transform_c = nn.Linear(1 * 2, common_size)

        
        self.fc1 = nn.Linear(common_size * 6, 256)
        self.dropout4 = nn.Dropout(0.5)  
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mva1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.dropout1(x)

       
        x = self.conv2(x)
        x = self.mva2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        
        x = self.conv3(x)
        x = self.mva3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.dropout3(x)

       
        x = self.conv4(x)
        x = self.mva4(x)
        x = F.relu(self.bn4(x))

        
        x = self.global_max_pool(x)
        x = x.squeeze(-1)  

        
        x_a, _ = self.lstm_a(x.unsqueeze(1))
        x_b, _ = self.lstm_b(x.unsqueeze(1))
        x_c, _ = self.lstm_c(x.unsqueeze(1))

        x_a = self.transform_a(x_a.squeeze(1))
        x_b = self.transform_b(x_b.squeeze(1))
        x_c = self.transform_c(x_c.squeeze(1))

        block1_features = torch.cat((x_a, x_b, x_c), dim=-1)

        
        ab = x_a * x_b
        ac = x_a * x_c
        bc = x_b * x_c

        block2_features = torch.cat((ab, ac, bc), dim=-1)

        
        higher_order_features = torch.cat((block1_features, block2_features), dim=-1).squeeze(1)
    
        x = F.relu(self.fc1(higher_order_features))
        x = self.dropout3(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x


ali = '/data1/ali/ppg_bg/'
file_path_normalized = os.path.join(ali, 'normalized_segments.npy')
file_path_bgv = os.path.join(ali, 'real_bgv.npy')
normalized_segments = np.load(file_path_normalized)
real_bgv = np.load(file_path_bgv)
fifteen_seconds_data_points = 15000
segments1 = [seg for seg in normalized_segments]
X = np.array(segments1).reshape(-1,1, fifteen_seconds_data_points)  
y = np.array(real_bgv)[:len(segments1)]

kf = KFold(n_splits=10, shuffle=True, random_state=8)

fold_results = []

for fold, (train_index, test_index) in enumerate(kf.split(segments1)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("Running Fold:", fold)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128)

    model = BloodGlucosePredictor()
    device = torch.device("cuda:3")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for inputs, targets in train_loader_tqdm:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loader_tqdm.set_postfix(train_loss=train_loss / len(train_loader_tqdm))

    # Testing phase
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            y_pred.extend(outputs.cpu().squeeze().tolist())
            y_true.extend(batch_y.cpu().tolist())

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    y_true_mgdl = y_true_np * 18
    y_pred_mgdl = y_pred_np * 18
    
    rmse = RMSE(y_pred_np, y_true_np)
    mard = MARD( y_pred_mgdl, y_true_mgdl)
    print(f"Fold {fold + 1}: RMSE={rmse:.2f}, MARD={mard:.2f}%")
    test_Clarke = zone_accuracy(y_true_mgdl, y_pred_mgdl, mode='clarke', detailed=False)
    A = test_Clarke[0]
    B = test_Clarke[1]
    C = test_Clarke[2]
    D = test_Clarke[3]
    E = test_Clarke[4]
    
    fold_results.append([fold+1, rmse, mard, A, B, C, D, E])
    
results_df = pd.DataFrame(fold_results, columns=['Fold', 'RMSE (mmol/L)', 'MARD (%)', 'CEG A (%)', 'CEG B (%)', 'CEG C (%)', 'CEG D (%)', 'CEG E (%)'])
avg_results = ['Average'] + [results_df[col].mean() for col in results_df.columns if col != 'Fold']
results_df.loc[len(results_df)] = avg_results  
print(results_df)
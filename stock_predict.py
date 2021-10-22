import numpy as np
import pandas as pd
import torch
torch.manual_seed(4)
from torch import nn, optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('bmh')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')
pin_memory = (device.type == 'cuda')


file_path = 'data/stock_aapl_5yr_20210926.xlsx'
n_seq = 5
original_data = pd.read_excel(file_path, sheet_name='x', usecols=['v','o','h','l','c'], dtype=np.single)
trimmed_data = np.zeros((original_data.shape[0]-n_seq+1, n_seq, original_data.shape[1]), dtype=np.single)
for i in range(n_seq-1):
	trimmed_data[:,i] = original_data.values[i:-(n_seq-i-1)]
trimmed_data[:,-1] = original_data.values[n_seq-1:]
trimmed_label = pd.read_excel(file_path, sheet_name='y', usecols=['y'], dtype=np.single).values[n_seq-1:]
print(f'Original Data Shape: {original_data.shape}, Trimmed Data Shape: {trimmed_data.shape}, Trimmed Label Shape: {trimmed_label.shape}')
original_data.drop(index=original_data.index, inplace=True)
train_set, test_set = train_test_split([(data, label) for data, label in zip(trimmed_data, trimmed_label)], test_size=50, shuffle=False)


class TorchDataset(torch.utils.data.Dataset):
	def __init__(self, dataset):
		super(TorchDataset, self).__init__()
		self.data = np.array([t[0] for t in dataset])
		self.label = np.array([t[1] for t in dataset])
	def __len__(self):
		return len(self.data)
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		data_batch = self.data[idx]
		data_batch = torch.from_numpy(data_batch)
		label_batch = self.label[idx]
		label_batch = torch.from_numpy(label_batch)
		return (data_batch, label_batch)


train_loader = torch.utils.data.DataLoader(TorchDataset(train_set), batch_size=1, shuffle=False, pin_memory=pin_memory)
test_loader = torch.utils.data.DataLoader(TorchDataset(test_set), batch_size=1, shuffle=False, pin_memory=pin_memory)


class TorchModel(nn.Module):
	def __init__(self):
		super(TorchModel, self).__init__()
		self.input_size = 5
		self.hidden_size = 5
		self.num_layers = 1
		self.hidden = torch.zeros((self.num_layers,1,self.hidden_size), dtype=torch.float32).to(device)
		self.cell = torch.zeros((self.num_layers,1,self.hidden_size), dtype=torch.float32).to(device)
		self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
		self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
		self.fc = nn.Linear(in_features=self.hidden_size, out_features=1)
		self.sig = nn.Sigmoid()
	def forward(self, x):
		# _, (self.hidden, self.cell) = self.lstm(x.squeeze(0).unsqueeze(1), (self.hidden, self.cell))  # LSTM
		_, self.hidden = self.gru(x.squeeze(0).unsqueeze(1), self.hidden)  # GRU
		# return self.hidden[-1].view(-1)  # direct output
		return self.fc(self.hidden[-1].view(-1))  # linear output
		# return self.sig(self.hidden[-1].view(-1))  # sigmoid output
		# return self.sig(self.fc(self.hidden[-1].view(-1)))  # linear => sigmoid output
	def reset_hidden(self):
		self.hidden = torch.zeros((self.num_layers,1,self.hidden_size), dtype=torch.float32).to(device)
		self.cell = torch.zeros((self.num_layers,1,self.hidden_size), dtype=torch.float32).to(device)


model = TorchModel()
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epoch = 10
start_epoch, end_epoch = 1, num_epoch


def train():
	model.train()
	print_interval = 200
	for epoch in range(start_epoch, end_epoch + 1):
		model.reset_hidden()
		for batch_idx, (data, label) in enumerate(train_loader):
			data = data.to(device)
			label = label.to(device)
			optimizer.zero_grad()
			y_pred = model(data)
			loss = criterion(y_pred, label.view(-1))
			loss.backward()
			optimizer.step()
			model.hidden.detach_()
			model.cell.detach_()
			if batch_idx > 0 and batch_idx % print_interval == 0:
				print(f'Train Epoch: {epoch} Progress: [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
		torch.cuda.empty_cache()
	return None


def test(dataset='testset'):
	model.eval()
	if dataset == 'trainset':
		model.reset_hidden()
		data_loader = train_loader
		y_pred_len = len(train_set)
	else:
		data_loader = test_loader
		y_pred_len = len(test_set)
	y_pred_arr = torch.zeros((y_pred_len,), dtype=torch.float32)
	with torch.no_grad():
		for batch_idx, (data, _) in enumerate(data_loader):
			data = data.to(device)
			y_pred = model(data)
			y_pred_arr[batch_idx] = y_pred
		torch.cuda.empty_cache()
	return y_pred_arr.numpy()


train()
train_pred = test('trainset')
test_pred = test('testset')


price_bottom, price_top = 26.02, 157.26
plt.plot(range(trimmed_label.shape[0]), trimmed_label.reshape(-1)*(price_top-price_bottom)+price_bottom, 'k-', label='True Label')
plt.plot(range(train_pred.shape[0]), train_pred*(price_top-price_bottom)+price_bottom, 'b-', label='Train Predict')
plt.plot(range(trimmed_label.shape[0]-test_pred.shape[0],trimmed_label.shape[0]), test_pred*(price_top-price_bottom)+price_bottom, 'y-', label='Test Predict')
plt.legend()
plt.title('Stock Prediction')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sqrtm import sqrtm


# class MyRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(MyRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.hidden_input=nn.Linear(input_size, hidden_size)
#         self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu')
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x, seq_lengths):
#         x=self.hidden_input(x)
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         # Padding: seq_length has to be 1D int64 cpu tensor x has to be float32 tensor
#         packed_seqs = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)
#         out, _ = self.rnn(packed_seqs, h0)
#         out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
#         out = out[:, -1, :]
#         out = self.fc(out)
#
#         return out

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers, output_size):
        super(MyRNN, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True, nonlinearity='tanh')
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        x = x.view(-1, input_dim)  # Reshape input to (batch_size*seq_len, input_dim)
        x = self.linear(x)
        x = x.view(batch_size, seq_len, -1)  # Reshape back to (batch_size, seq_len, hidden_size)
        output, _ = self.rnn(x)
        output = self.output_layer(output[:, -1, :])  # Take the last time-step output
        return output


# Model defining, load and output trial
model = MyRNN(2, 16, 3, 5)
model.load_state_dict(torch.load('trained_model_rnn_32-5.pth'))
# measurements_tt = torch.from_numpy(measurements).to(torch.float32)
# measurements_tt = measurements_tt.unsqueeze(0)
# measurements_tt_length = torch.tensor([measurements_tt.size(1)], dtype=torch.int64)
# output = model(measurements_tt)

def ws_dist(nn_output, label):
    nn_mid=torch.tensor([[(nn_output[0].item()), 0], [0, (nn_output[1].item())]])
    s = torch.sin(nn_output[4])
    c = torch.cos(nn_output[4])
    nn_rot = torch.stack([torch.stack([c, -s]),
                       torch.stack([s, c])])
    nn_mu=torch.tensor([[nn_output[2].item()],[nn_output[3].item()]])
    nn_cov_sqr=torch.mm(torch.mm(nn_rot.double(),nn_mid.double()),torch.transpose(nn_rot.double(),0,1))
    nn_cov=torch.mm(nn_cov_sqr,nn_cov_sqr)

    label_mid = torch.tensor([[(label[0].item()), 0], [0, (label[1].item())]])
    s = torch.sin(label[4])
    c = torch.cos(label[4])
    label_rot = torch.stack([torch.stack([c, -s]),
                          torch.stack([s, c])])
    label_mu = torch.tensor([[label[2].item()], [label[3].item()]])
    label_cov_sqr = torch.mm(torch.mm(label_rot.double(), label_mid.double()), torch.t(label_rot.double()))
    label_cov=torch.mm(label_cov_sqr,label_cov_sqr)

    dist=torch.sum(torch.square(nn_mu-label_mu))+torch.trace(nn_cov+label_cov-2*sqrtm(torch.mm(torch.mm(nn_cov_sqr, label_cov),nn_cov_sqr)))
    if torch.isnan(dist):
        print("loss=nan")
        exit()
    else:
        return dist

def ws_dist_batch(nn_output, label):
    batch_size=nn_output.size(0)
    distances=torch.zeros(batch_size,1)
    for k in range(batch_size):
        nn_mid = torch.tensor([[(nn_output[k][0].item()), 0], [0, (nn_output[k][1].item())]])
        s = torch.sin(nn_output[k][4])
        c = torch.cos(nn_output[k][4])
        nn_rot = torch.stack([torch.stack([c, -s]),
                               torch.stack([s, c])])
        nn_mu = torch.tensor([[nn_output[k][2].item()], [nn_output[k][3].item()]])
        nn_cov_sqr = torch.mm(torch.mm(nn_rot.double(), nn_mid.double()), torch.transpose(nn_rot.double(), 0, 1))
        nn_cov = torch.mm(nn_cov_sqr, nn_cov_sqr)

        label_mid = torch.tensor([[(label[k][0].item()), 0], [0, (label[k][1].item())]])
        s = torch.sin(label[k][4])
        c = torch.cos(label[k][4])
        label_rot = torch.stack([torch.stack([c, -s]),
                                 torch.stack([s, c])])
        label_mu = torch.tensor([[label[k][2].item()], [label[k][3].item()]])
        label_cov_sqr = torch.mm(torch.mm(label_rot.double(), label_mid.double()), torch.t(label_rot.double()))
        label_cov = torch.mm(label_cov_sqr, label_cov_sqr)

        dist = torch.sum(torch.square(nn_mu - label_mu)) + torch.trace(
            nn_cov + label_cov - 2 * sqrtm(torch.mm(torch.mm(nn_cov_sqr, label_cov), nn_cov_sqr)))
        distances[k]=dist
        mean=torch.mean(distances)
    return mean


optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.00001)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.MSELoss()
# Dataset Generation

'''
num_training_ellipses = 25
plt.figure()
dataset=torch.zeros([num_training_ellipses*4, 20],dtype=torch.float32)
labelset=torch.zeros([num_training_ellipses*4, 5], dtype=torch.float32)
for j in range(num_training_ellipses):
    print("ellipse number:", j+1)
    B = np.random.uniform(2, 5, 1)
    A = B + np.random.uniform(2, 5, 1)
    H = np.random.uniform(-2, 2, 1)
    K = np.random.uniform(-2, 2, 1)
    tau = 3.14 * np.random.uniform(0, 2, 1)
    measurement_num = int(np.random.uniform(50, 100, 1)[0])
    x_noise = np.random.uniform(0.05, 0.25, 1)
    y_noise = np.random.uniform(0.05, 0.25, 1)
    geo_parameters = [A/10, B/10 , H/4, K/4, tau/6.14]
    labels = np.array(geo_parameters)
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    label_tensor = label_tensor.squeeze(1)

    angles = np.linspace(0, 2 * np.pi, 100)
    x_ellipse = A * np.cos(angles) * np.cos(tau) - B * np.sin(angles) * np.sin(tau) + H
    y_ellipse = A * np.cos(angles) * np.sin(tau) + B * np.sin(angles) * np.cos(tau) + K


    # plt.plot(x_ellipse, y_ellipse, label='Train True Ellipse')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('True Ellipse')
    # plt.legend()
    # plt.axis('equal')
    # plt.grid(True)

    for k in range(4):
        random_angles = np.linspace(tau-np.pi/4, tau+np.pi/4, 10)+k*np.pi/2
        x_measurement = A * np.cos(random_angles) * np.cos(tau) - B * np.sin(random_angles) * np.sin(tau) + H
        y_measurement = A * np.cos(random_angles) * np.sin(tau) + B * np.sin(random_angles) * np.cos(tau) + K
        x_measurement_wnoise = x_measurement + np.random.normal(0, x_noise, x_measurement.shape)
        y_measurement_wnoise = y_measurement + np.random.normal(0, y_noise, y_measurement.shape)
        measurements = np.column_stack((x_measurement_wnoise, y_measurement_wnoise))
        # if k == 0:
        #     plt.scatter(x_measurement_wnoise, y_measurement_wnoise, color='red', label='Sampled Points up')
        # if k == 1:
        #     plt.scatter(x_measurement_wnoise, y_measurement_wnoise, color='blue', label='Sampled Points left')
        # if k == 2:
        #     plt.scatter(x_measurement_wnoise, y_measurement_wnoise, color='m', label='Sampled Points down')
        # if k == 3:
        #     plt.scatter(x_measurement_wnoise, y_measurement_wnoise, color='c', label='Sampled Points right')
        measurements_tt = torch.from_numpy(measurements).to(torch.float32)
        measurements_tt = measurements_tt.flatten()
        dataset[j*4+k]=measurements_tt
        labelset[j*4+k]=label_tensor
indices = torch.randperm(dataset.size()[0])
dataset=dataset[indices]
labelset=labelset[indices]
torch.save(dataset, 'v_data_90D.t')
torch.save(labelset, 'v_label_90D.t')
plt.show()

'''

# Training

datain=torch.load('data_rnn.t')
labelin=torch.load('label_rnn.t')
validate_data=torch.load('v_data_rnn.t')
validate_label=torch.load('v_label_rnn.t')

batch_size=1
loader=torch.utils.data.DataLoader(list(zip(datain, labelin)),shuffle=True, batch_size=batch_size)
val_loader=torch.utils.data.DataLoader(list(zip(validate_data, validate_label)),shuffle=False, batch_size=batch_size)

def weightedMSE(output,label):
    mean_all=0
    for l in range(len(output)):
        mean_s= (torch.square(output[l][0]-label[l][0])+torch.square(output[l][1]-label[l][1])
            +torch.square(output[l][2]-label[l][2])+torch.square(output[l][3]-label[l][3])
            +torch.square(output[l][4]-label[l][4])*5)/10
        mean_all=mean_all+mean_s
    mean=mean_all/len(output)
    return mean

def weightedMSEtensor(output, label):
    A=[]
    B=[]
    H=[]
    K=[]
    Tau=[]
    for l in range(len(output)):
        A.append((output[l][0]-label[l][0]).item()**2)
        B.append((output[l][1]-label[l][1]).item()**2)
        H.append((output[l][2]-label[l][2]).item()**2)
        K.append((output[l][3]-label[l][3]).item()**2)
        Tau.append((output[l][4]-label[l][4]).item()**2)
        tensor=[A,B,H,K,Tau]

    return tensor



epochs = 25
last_avg_loss=9999
val_losses=[]
train_losses=[]
decay=False
seq_length=torch.ones(1)*10
A_list=[]
B_list=[]
H_list=[]
K_list=[]
tau_list=[]
for epoch in range(epochs):
    print("epoch:",epoch)
    for batch, labelset in loader:
        optimizer.zero_grad()
        batch=batch.unflatten(1,(batch.size(1)//2,2))
        output=model(batch)
        label=labelset
        loss=weightedMSE(output,label)
        loss_tensor=weightedMSEtensor(output, label)
        A_list.extend(loss_tensor[0])
        B_list.extend(loss_tensor[1])
        H_list.extend(loss_tensor[2])
        K_list.extend(loss_tensor[3])
        tau_list.extend(loss_tensor[4])
        loss.backward()
        train_losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
    if epoch%5 == 0:
        avg_val_loss=0
        for validate_data, validate_label in val_loader:
            validate_data = validate_data.unflatten(1, (validate_data.size(1)//2, 2))
            output = model(validate_data)
            output = output
            label = validate_label
            val_loss = weightedMSE(output, label)
            avg_val_loss=avg_val_loss+val_loss
        val_losses.append(avg_val_loss.item())
        print("validation loss in epoch ", epoch, ":", avg_val_loss.item()/100)
        if last_avg_loss <= avg_val_loss/100:
            last_avg_loss = avg_val_loss/100
            print("loss decay")
            # decay = True
        else:
            last_avg_loss = avg_val_loss
            torch.save(model.state_dict(), 'trained_model_rnn_32-5.pth')
    if decay==True:
        break

plt.figure()
plt.plot(np.linspace(0, len(tau_list), len(tau_list)), tau_list, label='Tau')
plt.xlabel('Epoch')
plt.ylabel('tau')
plt.title('Tau')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(np.linspace(0, len(A_list), len(A_list)), A_list, label='A')
plt.xlabel('Epoch')
plt.ylabel('A')
plt.title('A')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(np.linspace(0, len(B_list), len(B_list)), B_list, label='B')
plt.xlabel('Epoch')
plt.ylabel('B')
plt.title('B')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(np.linspace(0, len(H_list), len(H_list)), H_list, label='H')
plt.xlabel('Epoch')
plt.ylabel('H')
plt.title('H')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(np.linspace(0, len(K_list), len(K_list)), K_list, label='K')
plt.xlabel('Epoch')
plt.ylabel('K')
plt.title('K')
plt.legend()
plt.grid(True)
plt.show()



plt.figure()
plt.plot(np.linspace(0, len(val_losses), len(val_losses)), val_losses, label='V_Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(np.linspace(0, len(train_losses), len(train_losses)), train_losses, label='T_Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()
plt.grid(True)
plt.show()

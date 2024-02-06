import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import scipy
from models import MyNN
from models import MyRNN
from models import MyLSTM
from models import StackedLSTM
from funcs import weightedMSEtensor
from funcs import ws_dist_batch
loss_fn=nn.MSELoss()


epochs = 11
NN_batchsize=32
#Train modes: NN, RNN , LSTM, SLSTM
train_mode='SLSTM'
load = 0
learning_rate = 1e-5
number_points = 10
last_avg_loss = 9999
val_losses = []
train_losses = []
A_list = []
B_list = []
H_list = []
K_list = []
tau_list = []
decay=False

class CustomWeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(CustomWeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, predicted, target):
        squared_errors = (predicted - target) ** 2
        weighted_errors = squared_errors * self.weights
        loss = torch.mean(weighted_errors)

        return loss
    
custom_weights = torch.tensor([1.0, 1.0, 1.0,1.0,5.0,1.0])
custom_loss = CustomWeightedMSELoss(custom_weights)


# NN trainer
if train_mode == 'NN':
    model = MyNN((number_points*2), 32, 6)
    model.train()
    if load == True:
        model.load_state_dict(torch.load('trained_model_nn.pth'))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
    datain = torch.load('data_nn.t')
    labelin = torch.load('label_nn.t')
    validate_data = torch.load('v_data_nn.t')
    validate_label = torch.load('v_label_nn.t')
    print(len(datain))

    loader = torch.utils.data.DataLoader(list(zip(datain, labelin)), shuffle=True, batch_size=NN_batchsize)
    val_loader = torch.utils.data.DataLoader(list(zip(validate_data, validate_label)), shuffle=False,
                                             batch_size=len(validate_data))

    for epoch in range(epochs):
        print("epoch:", epoch)
        for batch, labelset in loader:
            optimizer.zero_grad()
            output = model(batch)
            label = labelset
            loss = ws_dist_batch(output, label)
            loss_tensor = weightedMSEtensor(output, label)
            A_list.extend(loss_tensor[0])
            B_list.extend(loss_tensor[1])
            H_list.extend(loss_tensor[2])
            K_list.extend(loss_tensor[3])
            tau_list.extend(loss_tensor[4])
            loss.backward()
            train_losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        if epoch % 5 == 0:
            torch.save(model.state_dict(), 'trained_model_nn.pth')
            for validate_data, validate_label in val_loader:
                model.eval()
                output = model(validate_data)
                model.train()
                output = output.squeeze(0)
                label = validate_label.squeeze(0)
                val_loss = ws_dist_batch(output, label)
                scheduler.step(val_loss)
                print("validation loss in epoch ", epoch, ":", val_loss.item())
                val_losses.append(val_loss.item())
                np.savetxt("v_loss_nn.csv", val_losses, delimiter=", ", fmt='% s')
                np.savetxt("t_loss_nn.csv", train_losses, delimiter=", ", fmt='% s')
                np.savetxt("A_list_nn.csv",A_list,delimiter=", ",fmt='% s')
                np.savetxt("B_list_nn.csv",B_list,delimiter=", ",fmt='% s')
                np.savetxt("H_list_nn.csv",H_list,delimiter=", ",fmt='% s')
                np.savetxt("K_list_nn.csv",K_list,delimiter=", ",fmt='% s')
                np.savetxt("tau_list_nn.csv",tau_list,delimiter=", ",fmt='% s')
                if last_avg_loss <= val_loss:
                    last_avg_loss = val_loss
                    print("loss decay")
                    # decay = True
                    torch.save(model.state_dict(), 'trained_model_nn.pth')
                else:
                    last_avg_loss = val_loss
                    torch.save(model.state_dict(), 'trained_model_nn.pth')
        if decay == True:
            break
    torch.save(model.state_dict(), 'trained_model_nn.pth')

# RNN Trainer
if train_mode == 'RNN':
    model = MyRNN(2, 16, 3, 5)
    model.train()
    if load == True:
        model.load_state_dict(torch.load('trained_model_rnn.pth'))

    datain = torch.load('data_rnn.t')
    labelin = torch.load('label_rnn.t')
    validate_data = torch.load('v_data_rnn.t')
    validate_label = torch.load('v_label_rnn.t')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
    batch_size = 1
    loader = torch.utils.data.DataLoader(list(zip(datain, labelin)), shuffle=True, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(list(zip(validate_data, validate_label)), shuffle=False,
                                             batch_size=batch_size)

    decay = False
    seq_length = torch.ones(1) * 10
    for epoch in range(epochs):
        print("epoch:", epoch)
        for batch, labelset in loader:
            optimizer.zero_grad()
            batch = batch.unflatten(1, (batch.size(1) // 2, 2))
            output = model(batch)
            label = labelset
            loss = loss_fn(output, label)
            loss_tensor = weightedMSEtensor(output, label)
            A_list.extend(loss_tensor[0])
            B_list.extend(loss_tensor[1])
            H_list.extend(loss_tensor[2])
            K_list.extend(loss_tensor[3])
            tau_list.extend(loss_tensor[4])
            loss.backward()
            train_losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        if epoch % 5 == 0:
            avg_val_loss = 0
            for validate_data, validate_label in val_loader:
                model.eval()
                validate_data = validate_data.unflatten(1, (validate_data.size(1) // 2, 2))
                output = model(validate_data)
                model.train()
                label = validate_label
                val_loss = loss_fn(output, label)
                avg_val_loss = avg_val_loss + val_loss
            val_losses.append(avg_val_loss.item())
            scheduler.step(avg_val_loss)
            print("validation loss in epoch ", epoch, ":", avg_val_loss.item() / 100)
            np.savetxt("v_loss_rnn.csv", val_losses, delimiter=", ", fmt='% s')
            np.savetxt("t_loss_rnn.csv", train_losses, delimiter=", ", fmt='% s')
            np.savetxt("A_list_rnn.csv",A_list,delimiter=", ",fmt='% s')
            np.savetxt("B_list_rnn.csv",B_list,delimiter=", ",fmt='% s')
            np.savetxt("H_list_rnn.csv",H_list,delimiter=", ",fmt='% s')
            np.savetxt("K_list_rnn.csv",K_list,delimiter=", ",fmt='% s')
            np.savetxt("tau_list_rnn.csv",tau_list,delimiter=", ",fmt='% s')
            if last_avg_loss <= avg_val_loss / 100:
                last_avg_loss = avg_val_loss / 100
                print("loss decay")
                # decay = True
            else:
                last_avg_loss = avg_val_loss
                torch.save(model.state_dict(), 'trained_model_rnn.pth')
        if decay == True:
            break

if train_mode == 'LSTM':
    model = MyLSTM(2,64,2,6)
    model.train()
    if load == True:
        model.load_state_dict(torch.load('trained_model_LSTM.pth'))
    matfile = scipy.io.loadmat('data.mat')['data']
    datain = matfile[0]
    labelin = matfile[1]
    matfile = scipy.io.loadmat('validate_data.mat')['data']
    validate_data = matfile[0]
    validate_label = matfile[1]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
    batch_size = 1
    loader = torch.utils.data.DataLoader(list(zip(datain, labelin)), shuffle=True, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(list(zip(validate_data, validate_label)), shuffle=False,
                                             batch_size=batch_size)

    decay = False
    for epoch in range(epochs):
        print("epoch:", epoch)
        for batch, labelset in loader:
            optimizer.zero_grad()
            batch=batch.float()
            output = model(batch).float()
            label = labelset[0].float()
            loss = custom_loss(output, label)
            # loss_tensor = weightedMSEtensor(output, label)
            # A_list.extend(loss_tensor[0])
            # B_list.extend(loss_tensor[1])
            # H_list.extend(loss_tensor[2])
            # K_list.extend(loss_tensor[3])
            # tau_list.extend(loss_tensor[4])
            loss.backward()
            train_losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        if epoch % 5 == 0:
            avg_val_loss = 0
            for validate_data, validate_label in val_loader:
                model.eval()
                validate_data=validate_data.float()
                output = model(validate_data).float()
                model.train()
                label = validate_label[0].float()
                val_loss = loss_fn(output, label)
                avg_val_loss = avg_val_loss + val_loss
            val_losses.append(avg_val_loss.item())
            scheduler.step(avg_val_loss)
            print("validation loss in epoch ", epoch, ":", avg_val_loss.item() / 100)
            np.savetxt("v_loss_rnn.csv", val_losses, delimiter=", ", fmt='% s')
            np.savetxt("t_loss_rnn.csv", train_losses, delimiter=", ", fmt='% s')
            # np.savetxt("A_list_rnn.csv",A_list,delimiter=", ",fmt='% s')
            # np.savetxt("B_list_rnn.csv",B_list,delimiter=", ",fmt='% s')
            # np.savetxt("H_list_rnn.csv",H_list,delimiter=", ",fmt='% s')
            # np.savetxt("K_list_rnn.csv",K_list,delimiter=", ",fmt='% s')
            # np.savetxt("tau_list_rnn.csv",tau_list,delimiter=", ",fmt='% s')
            if last_avg_loss <= avg_val_loss / 100:
                last_avg_loss = avg_val_loss / 100
                print("loss decay")
                # decay = True
            else:
                last_avg_loss = avg_val_loss
                torch.save(model.state_dict(), 'trained_model_LSTM.pth')
        if decay == True:
            break


if train_mode == 'SLSTM':
    model = StackedLSTM(2,6,64,2,6)
    model.train()
    if load == True:
        model.load_state_dict(torch.load('trained_model_SLSTM.pth'))
    matfile = scipy.io.loadmat('Data_75.mat')['data']
    datain = matfile[0]
    labelin = matfile[1]
    prior = matfile[2]
    matfile = scipy.io.loadmat('Svalidate_data.mat')['data']
    validate_data = matfile[0]
    validate_label = matfile[1]
    val_prior = matfile[2]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
    batch_size = 1
    loader = torch.utils.data.DataLoader(list(zip(datain, labelin,prior)), shuffle=True, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(list(zip(validate_data, validate_label,val_prior)), shuffle=False,
                                             batch_size=batch_size)

    decay = False
    for epoch in range(epochs):
        print("epoch:", epoch)
        for batch, labelset,prior in loader:
            optimizer.zero_grad()
            batch=batch.float()
            input_prior=prior.float()
            output = model(batch,input_prior).float()
            label = labelset[0].float()
            loss = custom_loss(output, label)
            # loss_tensor = weightedMSEtensor(output, label)
            # A_list.extend(loss_tensor[0])
            # B_list.extend(loss_tensor[1])
            # H_list.extend(loss_tensor[2])
            # K_list.extend(loss_tensor[3])
            # tau_list.extend(loss_tensor[4])
            loss.backward()
            train_losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        if epoch % 5 == 0:
            avg_val_loss = 0
            for validate_data, validate_label,val_prior in val_loader:
                model.eval()
                validate_data=validate_data.float()
                val_input_prior=val_prior.float()
                output = model(validate_data,val_input_prior).float()
                model.train()
                label = validate_label[0].float()
                val_loss = loss_fn(output, label)
                avg_val_loss = avg_val_loss + val_loss
            val_losses.append(avg_val_loss.item())
            scheduler.step(avg_val_loss)
            print("validation loss in epoch ", epoch, ":", avg_val_loss.item() / 100)
            np.savetxt("v_loss_rnn.csv", val_losses, delimiter=", ", fmt='% s')
            np.savetxt("t_loss_rnn.csv", train_losses, delimiter=", ", fmt='% s')
            # np.savetxt("A_list_rnn.csv",A_list,delimiter=", ",fmt='% s')
            # np.savetxt("B_list_rnn.csv",B_list,delimiter=", ",fmt='% s')
            # np.savetxt("H_list_rnn.csv",H_list,delimiter=", ",fmt='% s')
            # np.savetxt("K_list_rnn.csv",K_list,delimiter=", ",fmt='% s')
            # np.savetxt("tau_list_rnn.csv",tau_list,delimiter=", ",fmt='% s')
            if last_avg_loss <= avg_val_loss / 100:
                last_avg_loss = avg_val_loss / 100
                print("loss decay")
                # decay = True
            else:
                last_avg_loss = avg_val_loss
                torch.save(model.state_dict(), 'trained_model_SLSTM.pth')
        if decay == True:
            break

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

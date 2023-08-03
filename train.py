import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sqrtm import sqrtm
from models import MyNN
from models import MyRNN
from models import LinearNet
from funcs import weightedMSE
from funcs import weightedMSEtensor
from funcs import ws_dist
from funcs import ws_dist_batch
from funcs import tau_loss
loss_fn=nn.MSELoss()


epochs = 50
NN_batchsize=32
#Train modes: NN, RNN
train_mode='RNN'
load = False
learning_rate=1e-3
number_points=5


last_avg_loss = 9999
val_losses = []
train_losses = []
A_list = []
B_list = []
H_list = []
K_list = []
tau_list = []
decay=False

# NN trainer
if train_mode == 'NN':
    model = MyNN(number_points*2, 32, 5)
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
            torch.save(model.state_dict(), 'trained_model_nn.pth')
            for validate_data, validate_label in val_loader:
                model.eval()
                output = model(validate_data)
                model.train()
                output = output.squeeze(0)
                label = validate_label.squeeze(0)
                val_loss = loss_fn(output, label)
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

# Output Plotting
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


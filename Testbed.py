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
from funcs import plot_losses
loss_fn=nn.MSELoss()


# Modes: NN , RNN
Test_Mode='RNN'

if Test_Mode == 'NN':
    model = MyNN(20, 32, 5)
    model.eval()
    v_loss = np.loadtxt("v_loss_nn.csv", delimiter=",", dtype=float)
    t_loss = np.loadtxt("t_loss_nn.csv", delimiter=",", dtype=float)
    A_list = np.loadtxt("A_list_nn.csv",delimiter=",", dtype=float)
    B_list = np.loadtxt("B_list_nn.csv",delimiter=",", dtype=float)
    H_list = np.loadtxt("H_list_nn.csv",delimiter=",", dtype=float)
    K_list = np.loadtxt("K_list_nn.csv",delimiter=",", dtype=float)
    tau_list = np.loadtxt("tau_list_nn.csv",delimiter=",", dtype=float)
    plot_losses(v_loss,t_loss,A_list,B_list,H_list,K_list,tau_list)
    model.load_state_dict(torch.load('trained_model_nn.pth'))
    model.eval()

    B = np.random.uniform(2, 5, 1)
    A = B + np.random.uniform(2, 5, 1)
    H = np.random.uniform(-2, 2, 1)
    K = np.random.uniform(-2, 2, 1)
    tau = 3.14 * np.random.uniform(0, 2, 1)
    measurement_num = int(np.random.uniform(50, 100, 1)[0])
    x_noise = np.random.uniform(0.05, 0.25, 1)
    y_noise = np.random.uniform(0.05, 0.25, 1)
    angles = np.linspace(0, 2*np.pi, 100)


    validate_data=torch.load('v_data_nn.t')
    validate_label=torch.load('v_label_nn.t')
    val_loader=torch.utils.data.DataLoader(list(zip(validate_data, validate_label)),shuffle=True, batch_size=2)
    for validate_data, validate_label in val_loader:
        label_counter=0
        output = model(validate_data)
        output = output.squeeze(0)
        label = validate_label.squeeze(0)
        geo_parameters = output.detach().numpy()
        for ellipse in geo_parameters:
            measurements=validate_data[label_counter].unflatten(0,(10,2))
            x_measurement_wnoise=measurements[:,0]
            y_measurement_wnoise=measurements[:,1]
            plt.figure()
            plt.scatter(x_measurement_wnoise, y_measurement_wnoise, color='red', label='Sampled Points')
            A = (label[label_counter][0]+1)*10
            B = (label[label_counter][1]+1)*10
            H = label[label_counter][2]*4
            K = label[label_counter][3]*4
            tau = label[label_counter][4]*3.14
            expanded_label=[A,B,H,K,tau]
            print("true label:", expanded_label)
            angles = np.linspace(0, 2 * np.pi, 100)
            x_ellipse = A * np.cos(angles) * np.cos(tau) - B * np.sin(angles) * np.sin(tau) + H
            y_ellipse = A * np.cos(angles) * np.sin(tau) + B * np.sin(angles) * np.cos(tau) + K
            plt.plot(x_ellipse, y_ellipse, label='Label Ellipse', color="m")
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Output Ellipse')
            plt.legend()
            plt.axis('equal')
            plt.grid(True)


            A = (ellipse[0]+1)*10
            B = (ellipse[1]+1)*10
            H = ellipse[2]*4
            K = ellipse[3]*4
            tau = ellipse[4]*3.14
            expanded_output=[A,B,H,K,tau]
            print("model output:", expanded_output)
            print("ws distance:", ws_dist(torch.tensor(expanded_output), torch.tensor(expanded_label)).item())
            print("MSE distance:", loss_fn(torch.tensor(expanded_output), torch.tensor(expanded_label)).item())
            label_counter = label_counter + 1
            angles = np.linspace(0, 2*np.pi, 100)
            x_ellipse = A * np.cos(angles) * np.cos(tau) - B * np.sin(angles) * np.sin(tau) + H
            y_ellipse = A * np.cos(angles) * np.sin(tau) + B * np.sin(angles) * np.cos(tau) + K
            plt.plot(x_ellipse, y_ellipse, label='Output Ellipse', color="yellow")
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Output Ellipse')
            plt.legend()
            plt.axis('equal')
            plt.grid(True)
            plt.show()



if Test_Mode == 'RNN':
    model = MyRNN(2, 16, 3, 5)
    model.eval()
    v_loss = np.loadtxt("v_loss_rnn.csv", delimiter=",", dtype=float)
    t_loss = np.loadtxt("v_loss_rnn.csv", delimiter=",", dtype=float)
    A_list = np.loadtxt("A_list_rnn.csv", delimiter=",", dtype=float)
    B_list = np.loadtxt("B_list_rnn.csv", delimiter=",", dtype=float)
    H_list = np.loadtxt("H_list_rnn.csv", delimiter=",", dtype=float)
    K_list = np.loadtxt("K_list_rnn.csv", delimiter=",", dtype=float)
    tau_list = np.loadtxt("tau_list_rnn.csv", delimiter=",", dtype=float)
    # plot_losses(v_loss,t_loss,A_list,B_list,H_list,K_list,tau_list)
    model.load_state_dict(torch.load('trained_model_rnn.pth'))
    model.eval()

    B = np.random.uniform(2, 5, 1)
    A = B + np.random.uniform(2, 5, 1)
    H = np.random.uniform(-2, 2, 1)
    K = np.random.uniform(-2, 2, 1)
    tau = 3.14 * np.random.uniform(0, 2, 1)
    measurement_num = int(np.random.uniform(50, 100, 1)[0])
    x_noise = np.random.uniform(0.05, 0.25, 1)
    y_noise = np.random.uniform(0.05, 0.25, 1)
    angles = np.linspace(0, 2 * np.pi, 100)

    validate_data = torch.load('v_data_rnn.t')
    validate_label = torch.load('v_label_rnn.t')
    batch_size = 1
    val_loader = torch.utils.data.DataLoader(list(zip(validate_data, validate_label)), shuffle=False,
                                             batch_size=batch_size)
    seq_length = torch.ones(1) * 10
    for validate_data, validate_label in val_loader:
        validate_data = validate_data.unflatten(1, (validate_data.size(1) // 2, 2))
        label_counter = 0
        output = model(validate_data)
        output = output.squeeze(0)
        label = validate_label[0]
        geo_parameters = output.detach().numpy()
        ellipse = geo_parameters
        measurements = validate_data[0]
        x_measurement_wnoise = measurements[:, 0]
        y_measurement_wnoise = measurements[:, 1]
        plt.figure()
        plt.scatter(x_measurement_wnoise, y_measurement_wnoise, color='red', label='Sampled Points')
        A = (label[0] + 1) * 10
        B = (label[1] + 1) * 10
        H = label[2] * 4
        K = label[3] * 4
        tau = label[4] * 3.14
        expanded_label = [A, B, H, K, tau]
        print("true label:", expanded_label)
        angles = np.linspace(0, 2 * np.pi, 100)
        x_ellipse = A * np.cos(angles) * np.cos(tau) - B * np.sin(angles) * np.sin(tau) + H
        y_ellipse = A * np.cos(angles) * np.sin(tau) + B * np.sin(angles) * np.cos(tau) + K
        plt.plot(x_ellipse, y_ellipse, label='Label Ellipse', color="m")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Output Ellipse')
        plt.legend()
        plt.axis('equal')
        # for i in range(len(x_measurement_wnoise)):
        #     plt.figure()
        #     plt.plot(x_ellipse, y_ellipse, label='Label Ellipse', color="m")
        #     plt.scatter(x_measurement_wnoise[i], y_measurement_wnoise[i], color='red', label='Sampled Points')
        #     plt.show()
        # exit()
        plt.grid(True)

        A = (ellipse[0] + 1) * 10
        B = (ellipse[1] + 1) * 10
        H = ellipse[2] * 4
        K = ellipse[3] * 4
        tau = ellipse[4] * 3.14
        expanded_output = [A, B, H, K, tau]
        print("model output:", expanded_output)
        print("ws distance:", ws_dist(torch.tensor(expanded_output), torch.tensor(expanded_label)).item())
        print("MSE distance:", loss_fn(torch.tensor(expanded_output), torch.tensor(expanded_label)).item())
        label_counter = label_counter + 1
        angles = np.linspace(0, 2 * np.pi, 100)
        x_ellipse = A * np.cos(angles) * np.cos(tau) - B * np.sin(angles) * np.sin(tau) + H
        y_ellipse = A * np.cos(angles) * np.sin(tau) + B * np.sin(angles) * np.cos(tau) + K
        plt.plot(x_ellipse, y_ellipse, label='Output Ellipse', color="yellow")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Output Ellipse')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()




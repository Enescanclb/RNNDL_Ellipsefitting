import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import scipy
from models import MyNN
from models import MyRNN
from models import MyLSTM
from models import StackedLSTM
from funcs import ws_dist
from funcs import plot_losses
from funcs import superellipse_sampler_1
from funcs import testbed_loss
import matplotlib.animation as animation
loss_fn=nn.MSELoss()


# Modes: NN , RNN, LSTM, SLSTM
Test_Mode='SLSTM'

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
        
if Test_Mode == 'LSTM':
    model = MyLSTM(2,64,2,6)
    model.eval()
    v_loss = np.loadtxt("v_loss_rnn.csv", delimiter=",", dtype=float)
    t_loss = np.loadtxt("t_loss_rnn.csv", delimiter=",", dtype=float)
    # A_list = np.loadtxt("A_list_rnn.csv", delimiter=",", dtype=float)
    # B_list = np.loadtxt("B_list_rnn.csv", delimiter=",", dtype=float)
    # H_list = np.loadtxt("H_list_rnn.csv", delimiter=",", dtype=float)
    # K_list = np.loadtxt("K_list_rnn.csv", delimiter=",", dtype=float)
    # tau_list = np.loadtxt("tau_list_rnn.csv", delimiter=",", dtype=float)
    # plot_losses(v_loss,t_loss,A_list,B_list,H_list,K_list,tau_list)
    model.load_state_dict(torch.load('trained_model_LSTM.pth'))
    mat = scipy.io.loadmat('validate_data.mat')
    data = mat["data"]
    validate_data = data[0]
    validate_label = data[1]
    batch_size = 1
    val_loader = torch.utils.data.DataLoader(list(zip(validate_data, validate_label)), shuffle=True,
                                             batch_size=batch_size)
    seq_length = torch.ones(1) * 10
    frames=[]
    for validate_data, validate_label in val_loader:
        frame=[]
        label_counter = 0
        output = model(validate_data.float())
        output = output.squeeze(0)
        label = validate_label[0][0]
        geo_parameters = output.detach().numpy()
        ellipse = geo_parameters
        measurements = validate_data[0]
        frame.append(measurements)
        # x_measurement_wnoise = measurements[:, 0]
        # y_measurement_wnoise = measurements[:, 1]
        # plt.figure()
        # plt.scatter(x_measurement_wnoise, y_measurement_wnoise, color='red', label='Sampled Points')
        A = (label[0].item()) * 10
        B = (label[1].item()) * 10
        H = (label[2].item())*50
        K = (label[3].item())*50
        tau = label[4].item()*np.pi
        q = label[5].item()*8 + 2
        expanded_label = [A, B, H, K, tau,q]
        frame.append(expanded_label)
        # print("true label:", expanded_label)
        # angles = np.linspace(0, 2 * np.pi, 100)
        # x_ellipse , y_ellipse = superellipse_sampler_1(A, B, H, K, tau, q, angles)
        # plt.plot(x_ellipse, y_ellipse, label='Label Ellipse', color="m")
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('Output Ellipse')
        # plt.legend()
        # plt.axis('equal')
        # for i in range(len(x_measurement_wnoise)):
        #     plt.figure()
        #     plt.plot(x_ellipse, y_ellipse, label='Label Ellipse', color="m")
        #     plt.scatter(x_measurement_wnoise[i], y_measurement_wnoise[i], color='red', label='Sampled Points')
        #     plt.show()
        # exit()
        # plt.grid(True)

        A = (ellipse[0].item()) *10
        B = (ellipse[1].item()) * 10
        H = (ellipse[2].item())*50
        K = (ellipse[3].item())*50
        tau = ellipse[4].item()*np.pi
        q = ellipse[5].item()*8 + 2
        expanded_output = [A, B, H, K, tau,q]
        frame.append(expanded_output)
        frames.append(frame)
        # print("model output:", expanded_output)
        # print("ws distance:", ws_dist(torch.tensor(expanded_output), torch.tensor(expanded_label)).item())
        # print("MSE distance:", loss_fn(torch.tensor(expanded_output), torch.tensor(expanded_label)).item())
        # label_counter = label_counter + 1
        # angles = np.linspace(0, 2 * np.pi, 100)
        # x_ellipse , y_ellipse = superellipse_sampler_1(A, B, H, K, tau, q, angles)
        # plt.plot(x_ellipse, y_ellipse, label='Output Ellipse', color="yellow")
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('Output Ellipse')
        # plt.legend()
        # plt.xlim((0,100));plt.ylim((0,100))
        # plt.grid(True)
        # plt.show()
        
if Test_Mode == 'SLSTM':
    model = StackedLSTM(2,6,64,2,6)
    model.eval()
    # v_loss = np.loadtxt("v_loss_rnn.csv", delimiter=",", dtype=float)
    # t_loss = np.loadtxt("t_loss_rnn.csv", delimiter=",", dtype=float)
    # A_list = np.loadtxt("A_list_rnn.csv", delimiter=",", dtype=float)
    # B_list = np.loadtxt("B_list_rnn.csv", delimiter=",", dtype=float)
    # H_list = np.loadtxt("H_list_rnn.csv", delimiter=",", dtype=float)
    # K_list = np.loadtxt("K_list_rnn.csv", delimiter=",", dtype=float)
    # tau_list = np.loadtxt("tau_list_rnn.csv", delimiter=",", dtype=float)
    # plot_losses(v_loss,t_loss,A_list,B_list,H_list,K_list,tau_list)
    model.load_state_dict(torch.load('trained_model_SLSTM.pth'))
    mat = scipy.io.loadmat('STurnScenario_Data.mat')
    data = mat["data"]
    validate_data = data[0]
    validate_label = data[1]
    prior = data[2]
    batch_size = 1
    val_loader = torch.utils.data.DataLoader(list(zip(validate_data, validate_label,prior)), shuffle=True,
                                             batch_size=batch_size)
    seq_length = torch.ones(1) * 10
    frames=[]
    losses=[]
    for validate_data, validate_label,prior in val_loader:
        frame=[]
        label_counter = 0
        output_ = model(validate_data.float(),prior.float())
        output_ = output_.squeeze(0)
        # output = model(validate_data.float(),output_.float())
        # output=output.squeeze(0)
        label = validate_label[0][0]
        # mid_step = output_.detach().numpy()
        geo_parameters = output_.detach().numpy()
        ellipse = geo_parameters
        measurements = validate_data[0]
        
        frame.append(measurements)
        A = (label[0].item()) * 10
        B = (label[1].item()) * 10
        H = (label[2].item())*75
        K = (label[3].item())*75
        tau = label[4].item()*np.pi
        q = label[5].item()*8 + 2
        expanded_label = [A, B, H, K, tau,q]
        frame.append(expanded_label)
        
        A = (ellipse[0].item()) *10
        B = (ellipse[1].item()) * 10
        H = (ellipse[2].item())*75
        K = (ellipse[3].item())*75
        tau = ellipse[4].item()*np.pi
        q = ellipse[5].item()*8 + 2
        expanded_output = [A, B, H, K, tau,q]
        frame.append(expanded_output)
        
        # A = (mid_step[0].item()) *10
        # B = (mid_step[1].item()) * 10
        # H = (mid_step[2].item())*75
        # K = (mid_step[3].item())*75
        # tau = mid_step[4].item()*np.pi
        # q = mid_step[5].item()*8 + 2
        # expanded_mid = [A, B, H, K, tau,q]
        
        fprior=prior[0][0]
        A = (fprior[0].item()) *10
        B = (fprior[1].item()) * 10
        H = (fprior[2].item())*75
        K = (fprior[3].item())*75
        tau = fprior[4].item()*np.pi
        q = fprior[5].item()*8 + 2
        expanded_prior = [A, B, H, K, tau,q]
        frame.append(expanded_prior)
        # frame.append(expanded_mid)
        frames.append(frame)
        loss_list = testbed_loss(expanded_output, expanded_label)
        losses.append(loss_list[6])


# plt.figure()
# plt.plot(np.linspace(0, len(v_loss), len(v_loss)), v_loss, label='V_Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss')
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.figure()
# plt.plot(np.linspace(0, len(t_loss), len(t_loss)), t_loss, label='T_Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss')
# plt.legend()
# plt.grid(True)
# plt.show()

plt.figure()
plt.hist(losses,50, label='TestLoss')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()
plt.grid(True)
plt.show()
print(sum(losses)/len(losses))

disp_length=100
fig, ax = plt.subplots(figsize=(13,13))
index=0
def update(frame):
    ax.clear()
    print("plotting")
    plt.xlim((-0,disp_length)),plt.ylim((-0,disp_length))
    plt.grid(True)
    measurements=frame[0]
    expanded_label=frame[1]
    expanded_output=frame[2]
    expanded_prior=frame[3]
    # expanded_mid=frame[4]
    x_measurement_wnoise = measurements[:, 0]
    y_measurement_wnoise = measurements[:, 1]
    ax.scatter(x_measurement_wnoise, y_measurement_wnoise, color='red', label='Sampled Points')

    A = expanded_label[0]
    B = expanded_label[1]
    H = expanded_label[2]
    K = expanded_label[3]
    tau = expanded_label[4]
    q = expanded_label[5]
    angles = np.linspace(0, 2 * np.pi, 100)
    x_ellipse , y_ellipse = superellipse_sampler_1(A, B, H, K, tau, q, angles)
    ax.plot(x_ellipse, y_ellipse, label='Label Ellipse', color="m")
    
    A = expanded_output[0]
    B = expanded_output[1]
    H = expanded_output[2]
    K = expanded_output[3]
    tau = expanded_output[4]
    q = expanded_output[5]
    x_ellipse , y_ellipse = superellipse_sampler_1(A, B, H, K, tau, q, angles)
    ax.plot(x_ellipse, y_ellipse, label='Output Ellipse', color="green")
    
    # A = expanded_mid[0]
    # B = expanded_mid[1]
    # H = expanded_mid[2]
    # K = expanded_mid[3]
    # tau = expanded_mid[4]
    # q = expanded_mid[5]
    # angles = np.linspace(0, 2 * np.pi, 100)
    # x_ellipse , y_ellipse = superellipse_sampler_1(A, B, H, K, tau, q, angles)
    # ax.plot(x_ellipse, y_ellipse, label='Mid Step', color="b")
    
    A = expanded_prior[0]
    B = expanded_prior[1]
    H = expanded_prior[2]
    K = expanded_prior[3]
    tau = expanded_prior[4]
    q = expanded_prior[5]
    x_ellipse , y_ellipse = superellipse_sampler_1(A, B, H, K, tau, q, angles)
    ax.plot(x_ellipse, y_ellipse, label='Prior Ellipse', color="cyan")
    
    plt.title('Output Ellipse')
    ax.legend()
    ax.set_xlim((0,75));ax.set_ylim((0,75))
    plt.grid(True)


anim = animation.FuncAnimation(fig, update, frames=frames, repeat=False)
anim.save("ellipse_fit_nn.mp4",fps=1)



import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import scipy
from models import StackedLSTM
from models import MyLSTM
from funcs import superellipse_sampler_1
from funcs import testbed_loss
import matplotlib.animation as animation
import time
start_time = time.time()
loss_fn=nn.MSELoss()

#Modes: SLSTM, LSTM
Test_mode="SLSTM"
if Test_mode == 'SLSTM':
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
    initial_prior=torch.from_numpy(data[2][0]).unsqueeze(0)
    batch_size = 1
    val_loader = torch.utils.data.DataLoader(list(zip(validate_data, validate_label)), shuffle=False,
                                             batch_size=batch_size)
    seq_length = torch.ones(1) * 10
    frames=[]
    losses=[]
    prior=initial_prior
    prev_output=initial_prior
    for validate_data, validate_label in val_loader:
        frame=[]
        label_counter = 0
        output = model(validate_data.float(),prior.float())
        output = output.squeeze(0)
        label = validate_label[0][0]
        geo_parameters = output.detach().numpy()
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
        fprior=prior[0][0]
        time_updated=output
        A = (fprior[0].item()) *10
        B = (fprior[1].item()) * 10
        H = (fprior[2].item())*75
        K = (fprior[3].item())*75
        tau = fprior[4].item()*np.pi
        q = fprior[5].item()*8 + 2
        expanded_prior = [A, B, H, K, tau,q]
        prev_output=expanded_output
        Delta_H=expanded_output[2]-prev_output[2]
        Delta_K=expanded_output[3]-prev_output[3]
        expanded_new_prior=[expanded_output[0],expanded_output[1],expanded_output[2],expanded_output[3],expanded_output[4],expanded_output[5]]
        prior[0][0][0]=expanded_new_prior[0]/10
        prior[0][0][1]=expanded_new_prior[1]/10
        prior[0][0][2]=expanded_new_prior[2]/75
        prior[0][0][3]=expanded_new_prior[3]/75
        prior[0][0][4]=expanded_new_prior[4]/np.pi
        prior[0][0][5]=(expanded_new_prior[5]-2)/8
        prev_output=expanded_output
        frame.append(expanded_prior)
        frames.append(frame)
        loss_list = testbed_loss(expanded_output, expanded_label)
        losses.append(loss_list[6])
        
        
if Test_mode == 'LSTM':
    model = MyLSTM(2,64,2,6)
    model.eval()
    # v_loss = np.loadtxt("v_loss_rnn.csv", delimiter=",", dtype=float)
    # t_loss = np.loadtxt("t_loss_rnn.csv", delimiter=",", dtype=float)
    # A_list = np.loadtxt("A_list_rnn.csv", delimiter=",", dtype=float)
    # B_list = np.loadtxt("B_list_rnn.csv", delimiter=",", dtype=float)
    # H_list = np.loadtxt("H_list_rnn.csv", delimiter=",", dtype=float)
    # K_list = np.loadtxt("K_list_rnn.csv", delimiter=",", dtype=float)
    # tau_list = np.loadtxt("tau_list_rnn.csv", delimiter=",", dtype=float)
    # plot_losses(v_loss,t_loss,A_list,B_list,H_list,K_list,tau_list)
    model.load_state_dict(torch.load('trained_model_LSTM.pth'))
    mat = scipy.io.loadmat('STurnScenario_Data.mat')
    data = mat["data"]
    validate_data = data[0]
    validate_label = data[1]
    initial_prior=torch.from_numpy(data[2][0]).unsqueeze(0)
    batch_size = 1
    val_loader = torch.utils.data.DataLoader(list(zip(validate_data, validate_label)), shuffle=False,
                                             batch_size=batch_size)
    seq_length = torch.ones(1) * 10
    frames=[]
    losses=[]
    prior=initial_prior
    prev_output=initial_prior
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
        fprior=prior[0][0]
        time_updated=output
        A = (fprior[0].item()) *10
        B = (fprior[1].item()) * 10
        H = (fprior[2].item())*75
        K = (fprior[3].item())*75
        tau = fprior[4].item()*np.pi
        q = fprior[5].item()*8 + 2
        expanded_prior = [A, B, H, K, tau,q]
        prev_output=expanded_output
        Delta_H=expanded_output[2]-prev_output[2]
        Delta_K=expanded_output[3]-prev_output[3]
        expanded_new_prior=[expanded_output[0],expanded_output[1],expanded_output[2],expanded_output[3],expanded_output[4],expanded_output[5]]
        prior[0][0][0]=expanded_new_prior[0]/10
        prior[0][0][1]=expanded_new_prior[1]/10
        prior[0][0][2]=expanded_new_prior[2]/75
        prior[0][0][3]=expanded_new_prior[3]/75
        prior[0][0][4]=expanded_new_prior[4]/np.pi
        prior[0][0][5]=(expanded_new_prior[5]-2)/8
        prev_output=expanded_output
        frame.append(expanded_prior)
        frames.append(frame)
        loss_list = testbed_loss(expanded_output, expanded_label)
        losses.append(loss_list[6])


print("--- %s seconds ---" % (time.time() - start_time))
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
anim.save("dynamictest.mp4",fps=1)
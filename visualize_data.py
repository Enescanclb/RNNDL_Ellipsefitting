import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy
from funcs import superellipse_sampler_1
matfile = scipy.io.loadmat('Svalidate_data.mat')['data']
datain = matfile[0]
labelin = matfile[1]
prior = matfile[2]
loader = torch.utils.data.DataLoader(list(zip(datain, labelin,prior)), shuffle=False, batch_size=1)

for validate_data, validate_label,val_prior in loader:
    label = validate_label[0][0].detach()
    prior = val_prior[0][0].detach()
    measurements = validate_data[0].detach()
    x_measurement_wnoise = measurements[:, 0]
    y_measurement_wnoise = measurements[:, 1]
    plt.figure()
    plt.scatter(x_measurement_wnoise, y_measurement_wnoise, color='red', label='Sampled Points')
    A = (label[0].item()) * 10
    B = (label[1].item()) * 10
    H = label[2].item() * 50
    K = label[3].item() * 50
    tau = label[4].item() * 3.14
    q = label[5].item() * 8 + 2
    expanded_label = [A, B, H, K, tau]
    angles = np.linspace(0, 2 * np.pi, 100)
    x_ellipse, y_ellipse = superellipse_sampler_1(A, B, H, K, tau, q, angles)
    plt.plot(x_ellipse, y_ellipse, label='Label Ellipse', color="m")
    A = (prior[0].item()) * 10
    B = (prior[1].item()) * 10
    H = prior[2].item() * 50
    K = prior[3].item() * 50
    tau = prior[4].item() * 3.14
    q = prior[5].item() * 8 + 2
    expanded_prior = [A, B, H, K, tau,q]
    angles = np.linspace(0, 2 * np.pi, 100)
    x_ellipse, y_ellipse = superellipse_sampler_1(A, B, H, K, tau, q, angles)
    plt.plot(x_ellipse, y_ellipse, label='Prior Ellipse', color="c")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Output Ellipse')
    plt.legend()
    plt.xlim((0,100));plt.ylim((0,100))
    plt.grid(True)
    plt.show()
    plt.pause(0.5)
    plt.close('all')
    # print("rr")
import numpy as np
import matplotlib.pyplot as plt
import torch

# read data_nn.t
# read label_nn.t

datain = torch.load('data_nn.t')
labelin = torch.load('label_nn.t')
print(len(datain))
loader = torch.utils.data.DataLoader(list(zip(datain, labelin)), shuffle=True, batch_size=1)

for validate_data, validate_label in loader:
    validate_data = validate_data.unflatten(1, (validate_data.size(1) // 2, 2))
    label_counter = 0
    label = validate_label[0]
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
    plt.grid(True)
    plt.show()
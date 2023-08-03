import numpy as np
import matplotlib.pyplot as plt
import torch


num_training_ellipses = 25

plt.figure()
# dataset = torch.zeros([num_training_ellipses * 4], dtype=torch.float32)
# labelset = torch.zeros([num_training_ellipses * 4], dtype=torch.float32)
# For rnn data
dataset=[]
labelset=[]
for j in range(num_training_ellipses):
    print("ellipse number:", j+1)
    B = np.random.uniform(2, 5, 1)
    A = B + np.random.uniform(2, 5, 1)
    H = np.random.uniform(-2, 2, 1)
    K = np.random.uniform(-2, 2, 1)
    tau = np.pi * np.random.uniform(-1, 1, 1)/2
    measurement_num = int(np.random.uniform(25, 75, 1)[0])
    measurement_num = 5

    y_noise = np.random.uniform(0.05, 0.25, 1)
    x_noise = np.random.uniform(0.05, 0.25, 1)
    geo_parameters = [A/10-1, B/10-1, H/4, K/4, tau/3.14]
    labels = np.array(geo_parameters)
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    label_tensor = label_tensor.squeeze(1)

    angles = np.linspace(0, 1 * np.pi, 100)
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
        random_angles = np.linspace(tau-np.pi/4, tau+np.pi/4, measurement_num)+k*np.pi/2
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
        # dataset[j*4+k]=measurements_tt
        # labelset[j*4+k]=label_tensor
        #For rnn data
        dataset.append(measurements_tt)
        labelset.append(label_tensor)

torch.save(dataset, 'v_data_nn.t')
torch.save(labelset, 'v_label_nn.t')

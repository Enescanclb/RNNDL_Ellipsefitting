import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class MyNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyNN, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.leakyrelu1 = nn.LeakyReLU(0.2)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.leakyrelu2 = nn.LeakyReLU(0.2)
        self.hidden3 = nn.Linear(hidden_size, hidden_size)
        self.leakyrelu3 = nn.LeakyReLU(0.2)
        self.hidden4 = nn.Linear(hidden_size, hidden_size)
        self.leakyrelu4 = nn.LeakyReLU(0.2)
        self.hidden5 = nn.Linear(hidden_size, hidden_size)
        self.leakyrelu5 = nn.LeakyReLU(0.2)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.leakyrelu1(x)
        x = self.hidden2(x)
        x = self.leakyrelu2(x)
        x = self.hidden3(x)
        x = self.leakyrelu3(x)
        x = self.hidden4(x)
        x = self.leakyrelu4(x)
        x = self.hidden5(x)
        x = self.leakyrelu5(x)
        x = self.output_layer(x)
        return x

model = MyNN(10, 32, 5)
model.load_state_dict(torch.load('trained_model_nn_32-5.pth'))

B = np.random.uniform(2, 5, 1)
A = B + np.random.uniform(2, 5, 1)
H = np.random.uniform(-2, 2, 1)
K = np.random.uniform(-2, 2, 1)
tau = 3.14 * np.random.uniform(0, 2, 1)
measurement_num = int(np.random.uniform(50, 100, 1)[0])
x_noise = np.random.uniform(0.05, 0.25, 1)
y_noise = np.random.uniform(0.05, 0.25, 1)
angles = np.linspace(0, 2*np.pi, 100)

x_ellipse = A * np.cos(angles) * np.cos(tau) - B * np.sin(angles) * np.sin(tau) + H
y_ellipse = A * np.cos(angles) * np.sin(tau) + B * np.sin(angles) * np.cos(tau) + K
random_angles = np.linspace(tau-np.pi/9, tau+np.pi/9, 5)+np.random.randint(3)*np.pi/2
x_measurement = A * np.cos(random_angles) * np.cos(tau) - B * np.sin(random_angles) * np.sin(tau) + H
y_measurement = A * np.cos(random_angles) * np.sin(tau) + B * np.sin(random_angles) * np.cos(tau) + K
x_measurement_wnoise = x_measurement+np.random.normal(0, y_noise, x_measurement.shape)
y_measurement_wnoise = y_measurement+np.random.normal(0, x_noise, y_measurement.shape)
new_measurements = np.column_stack((x_measurement_wnoise, y_measurement_wnoise))

plt.figure()
plt.plot(x_ellipse, y_ellipse, label='Test True Ellipse', color="m")
plt.scatter(x_measurement_wnoise, y_measurement_wnoise, color='blue', label='Sampled Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('True Ellipse')
plt.legend()
plt.axis('equal')
plt.grid(True)

measurements_tt = torch.from_numpy(new_measurements).to(torch.float32)
measurements_tt = measurements_tt.flatten()
output = model(measurements_tt)
print(output.detach())
geo_parameters = output.detach().numpy()
A = geo_parameters[0]
B = geo_parameters[1]
H = geo_parameters[2]
K = geo_parameters[3]
tau = [4]
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
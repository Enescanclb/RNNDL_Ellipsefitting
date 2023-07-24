import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


"abcdef parameters sampling"
# def implicit_eq(x, y , teta):
#     a = teta[0]
#     b = teta[1]
#     c = teta[2]
#     d = teta[3]
#     e = teta[4]
#     f = teta[5]
#     return a*x**2 + 2*b*x*y + c*y**2 + 2*d*x + 2*e*y +f
#
# a = 3
# b = -2
# c = 3
# d = 0
# e = 0
# f = -1
# teta=(a,b,c,d,e,f)
#
# x = np.linspace(-10, 10, 100)
# y = np.linspace(-10, 10, 100)
# X, Y = np.meshgrid(x, y)
# Z = implicit_eq(X, Y,teta)
# contour = plt.contour(X, Y, Z, levels=[0], colors='y')
# plt.grid(True)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Implicit Equation')
# points = contour.collections[0].get_paths()[0].vertices
# x_coords = points[:, 0]
# y_coords = points[:, 1]
# noise=0.1
# x_noise=x_coords+np.random.normal(0,noise,x_coords.shape)
# y_noise=y_coords+np.random.normal(0,noise,y_coords.shape)
# measurements=np.column_stack((x_noise,y_noise))
# labels=teta
# #plt.plot(measurements[:, 0], measurements[:, 1], 'ro')
# plt.show()
"algebraic to geometric function"
# def algebraic_to_geometric(alg_params):
#     a = alg_params[0]
#     b = alg_params[1]
#     c = alg_params[2]
#     d = alg_params[3]
#     e = alg_params[4]
#     f = alg_params[5]
#
#     # Calculate Delta and lambda_plus, lambda_minus
#     Delta = b**2 - 4 * a * c
#     lambda_plus = 0.5 * (a + c - np.sqrt(b**2 + (a-c)**2))
#     lambda_minus = 0.5 * (a + c + np.sqrt(b**2 + (a-c)**2))
#
#     psi = b*d*e - a*e*e - b*b*f + c*(4*a*f - d**2)
#
#     V_plus = np.sqrt(psi / (lambda_plus * Delta))
#     V_minus = np.sqrt(psi / (lambda_minus * Delta))
#
#     A = max(V_plus, V_minus)
#     B = min(V_plus, V_minus)
#
#     H = (2*c*d - b*e) / Delta
#     K = (2*a*e - b*d) / Delta
#
#     if V_plus > V_minus:
#
#         if b < 0:
#
#             if a < c:
#                 tau = 0.5 * np.arctan(((a - c) / b)**(-1))
#             elif a == c:
#                 tau = np.pi/4
#             else:
#                 tau = 0.5 * np.arctan(((a - c) / b)**(-1)) + np.pi/2
#
#         elif b == 0:
#
#             if a < c:
#                 tau = 0
#             else:
#                 tau = np.pi/2
#
#         else:
#
#             if a < c:
#                 tau = 0.5 * np.arctan((a - c) / b) + np.pi
#             elif a == c:
#                 tau = 3*np.pi/4
#             else:
#                 tau = 0.5 * np.arccot((a - c) / b) + np.pi/2
#
#     elif V_plus < V_minus:
#
#         if b < 0:
#
#             if a < c:
#                 tau = 0.5 * np.arccot((a - c) / b) + np.pi/2
#             elif a == c:
#                 tau = 3*np.pi/4
#             else:
#                 tau = 0.5 * np.arccot((a - c) / b) + np.pi
#
#         elif b == 0:
#
#             if a < c:
#                 tau = np.pi/2
#             else:
#                 tau = 0
#
#         else:
#
#             if a < c:
#                 tau = 0.5 * np.arccot((a - c) / b) + np.pi/2
#             elif a == c:
#                 tau = np.pi/4
#             else:
#                 tau = 0.5 * np.arccot((a - c) / b)
#
#     else:
#         tau = 0
#
#     geo_params = {}
#     geo_params['A'] = A
#     geo_params['B'] = B
#     geo_params['H'] = H
#     geo_params['K'] = K
#     geo_params['tau'] = tau
#     geo_params['W'] = np.dot(np.dot(np.array([[np.cos(tau), -np.sin(tau)], [np.sin(tau), np.cos(tau)]]), np.diag([A**2, B**2])), np.array([[np.cos(tau), np.sin(tau)], [-np.sin(tau), np.cos(tau)]]))
#     geo_params['c'] = np.array([H, K])
#     geo_params['unit2ellipse'] = lambda x: np.dot(np.dot(np.array([[np.cos(tau), -np.sin(tau)], [np.sin(tau), np.cos(tau)]]), np.diag([A, B])), x) + np.array([H, K])
#     geo_params['ellipse2unit'] = lambda x: np.dot(np.dot(np.array([[1/A, 0], [0, 1/B]]), np.array([[np.cos(tau), np.sin(tau)], [-np.sin(tau), np.cos(tau)]])), (x - np.array([H, K])))
#
#     return geo_params

B = np.random.uniform(0, 5, 1)
A = B+np.random.uniform(0, 5, 1)
H = np.random.uniform(-2, 2, 1)
K = np.random.uniform(-2, 2, 1)
tau = 3.14*np.random.uniform(0, 2, 1)
num_measurement = np.random.uniform(50, 100, 1)
x_noise = np.random.uniform(0.05, 0.25, 1)
y_noise = np.random.uniform(0.05, 0.25, 1)
geo_parameters = [A, B, H, K, tau]
labels = np.array(geo_parameters)
label_tensor = torch.tensor(labels, dtype=torch.float32)

angles = np.linspace(0, 2*np.pi, 100)
x_ellipse = A * np.cos(angles) * np.cos(tau) - B * np.sin(angles) * np.sin(tau) + H
y_ellipse = A * np.cos(angles) * np.sin(tau) + B * np.sin(angles) * np.cos(tau) + K
measurement_num = 20
random_angles = np.random.rand(measurement_num) * 2*np.pi
x_measurement = A * np.cos(random_angles) * np.cos(tau) - B * np.sin(random_angles) * np.sin(tau) + H
y_measurement = A * np.cos(random_angles) * np.sin(tau) + B * np.sin(random_angles) * np.cos(tau) + K
x_measurement_wnoise = x_measurement+np.random.normal(0, x_noise, x_measurement.shape)
y_measurement_wnoise = y_measurement+np.random.normal(0, y_noise, y_measurement.shape)
measurements = np.column_stack((x_measurement_wnoise, y_measurement_wnoise))


# plt.figure()
# plt.plot(x_ellipse, y_ellipse, label='Train True Ellipse')
# plt.scatter(x_measurement_wnoise, y_measurement_wnoise, color='red', label='Sampled Points')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('True Ellipse')
# plt.legend()
# plt.axis('equal')
# plt.grid(True)
# #plt.show()


class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MyRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, seq_lengths):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Padding: seq_length has to be 1D int64 cpu tensor x has to be float32 tensor
        packed_seqs = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)
        out, _ = self.rnn(packed_seqs, h0)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = out[:, -1, :]
        out = self.fc(out)

        return out

# Model defining, load and output trial
model = MyRNN(2, 16, 1, 5)
model.load_state_dict(torch.load('trained_model_16-1.pth'))
measurements_tt = torch.from_numpy(measurements).to(torch.float32)
measurements_tt = measurements_tt.unsqueeze(0)
measurements_tt_length = torch.tensor([measurements_tt.size(1)], dtype=torch.int64)
output = model(measurements_tt, measurements_tt_length)


def sine_loss(nn_output, label):
    return torch.sin(nn_output - label).square().mean()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)


# Training
num_training_ellipses = 5
num_permutations = 50
plt.figure()
for j in range(num_training_ellipses):
    print("ellipse number:", j+1)
    B = np.random.uniform(0, 5, 1)
    A = B + np.random.uniform(0, 5, 1)
    H = np.random.uniform(-2, 2, 1)
    K = np.random.uniform(-2, 2, 1)
    tau = 3.14 * np.random.uniform(0, 2, 1)
    measurement_num = int(np.random.uniform(50, 100, 1)[0])
    x_noise = np.random.uniform(0.05, 0.25, 1)
    y_noise = np.random.uniform(0.05, 0.25, 1)
    geo_parameters = [A, B, H, K, tau]
    labels = np.array(geo_parameters)
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    label_tensor = label_tensor.squeeze(1)

    angles = np.linspace(0, 2 * np.pi, 100)
    x_ellipse = A * np.cos(angles) * np.cos(tau) - B * np.sin(angles) * np.sin(tau) + H
    y_ellipse = A * np.cos(angles) * np.sin(tau) + B * np.sin(angles) * np.cos(tau) + K
    random_angles = np.random.rand(measurement_num) * 2 * np.pi
    x_measurement = A * np.cos(random_angles) * np.cos(tau) - B * np.sin(random_angles) * np.sin(tau) + H
    y_measurement = A * np.cos(random_angles) * np.sin(tau) + B * np.sin(random_angles) * np.cos(tau) + K
    x_measurement_wnoise = x_measurement + np.random.normal(0, x_noise, x_measurement.shape)
    y_measurement_wnoise = y_measurement + np.random.normal(0, y_noise, y_measurement.shape)
    measurements = np.column_stack((x_measurement_wnoise, y_measurement_wnoise))

    plt.plot(x_ellipse, y_ellipse, label='Train True Ellipse')
    # plt.scatter(x_measurement_wnoise, y_measurement_wnoise, color='red', label='Sampled Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('True Ellipse')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)

    for k in range(num_permutations):
        print("permutation:", k + 1)
        np.random.shuffle(measurements)
        measurements_tt = torch.from_numpy(measurements).to(torch.float32)
        measurements_tt = measurements_tt.unsqueeze(0)
        measurements_tt_length = torch.tensor([measurements_tt.size(1)], dtype=torch.int64)
        epochs = 1000
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(measurements_tt, measurements_tt_length)
            output = output.squeeze(0)
            # choose loss function: sine_loss is not really reliable
            # loss=sine_loss(output, label_tensor)
            loss = nn.MSELoss()(output, label_tensor)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(loss.item())

            if loss == 0:
                print("loss is 0")
                break

    torch.save(model.state_dict(), 'trained_model_16-1.pth')
plt.show()



"abcdef output testing"
# a = 5
# b = -2
# d = 1
# e = 0.2
# f = 4
# teta=(a,b,d,e,f)
#
# x = np.linspace(-10, 10, 100)
# y = np.linspace(-10, 10, 100)
# X, Y = np.meshgrid(x, y)
# Z = implicit_eq(X, Y,teta)
# contour = plt.contour(X, Y, Z, levels=[0], colors='b')
# plt.grid(True)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Fitted Ellipse')
# points = contour.collections[0].get_paths()[0].vertices
# x_coords = points[:, 0]
# y_coords = points[:, 1]
# noise=0.1
# x_noise=x_coords+np.random.normal(0,noise,x_coords.shape)
# y_noise=y_coords+np.random.normal(0,noise,y_coords.shape)
# measurements=np.column_stack((x_noise,y_noise))
#
# measurements_tt = torch.from_numpy(measurements).to(torch.float32)
# measurements_tt = measurements_tt.unsqueeze(0)
# measurements_tt_length = torch.tensor([measurements_tt.size(1)], dtype=torch.int64)
#
# out_teta=torch.t(model(measurements_tt,measurements_tt_length)).detach().numpy()
# x = np.linspace(-10, 10, 100)
# y = np.linspace(-10, 10, 100)
# X, Y = np.meshgrid(x, y)
# Z = implicit_eq(X, Y,out_teta)
# contour = plt.contour(X, Y, Z, levels=[0], colors='m')
# plt.plot(measurements[:, 0], measurements[:, 1], 'ro')
# plt.show()

B = np.random.uniform(0, 5, 1)
A = B + np.random.uniform(0, 5, 1)
H = np.random.uniform(-2, 2, 1)
K = np.random.uniform(-2, 2, 1)
tau = 3.14 * np.random.uniform(0, 2, 1)
measurement_num = int(np.random.uniform(50, 100, 1)[0])
x_noise = np.random.uniform(0.05, 0.25, 1)
y_noise = np.random.uniform(0.05, 0.25, 1)
angles = np.linspace(0, 2*np.pi, 100)

x_ellipse = A * np.cos(angles) * np.cos(tau) - B * np.sin(angles) * np.sin(tau) + H
y_ellipse = A * np.cos(angles) * np.sin(tau) + B * np.sin(angles) * np.cos(tau) + K
random_angles = np.random.rand(measurement_num) * 2*np.pi
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
measurements_tt = measurements_tt.unsqueeze(0)
measurements_tt_length = torch.tensor([measurements_tt.size(1)], dtype=torch.int64)
output = model(measurements_tt, measurements_tt_length)
geo_parameters = output.detach().numpy()[0]
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

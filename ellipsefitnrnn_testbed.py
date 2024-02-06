import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sqrtm import sqrtm

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers, output_size):
        super(MyRNN, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu')
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        x = x.view(-1, input_dim)  # Reshape input to (batch_size*seq_len, input_dim)
        x = self.linear(x)
        x = x.view(batch_size, seq_len, -1)  # Reshape back to (batch_size, seq_len, hidden_size)
        output, _ = self.rnn(x)
        output = self.output_layer(output[:, -1, :])  # Take the last time-step output
        return output

def ws_dist(nn_output, label):
    nn_mid=torch.tensor([[(nn_output[0].item()), 0], [0, (nn_output[1].item())]])
    s = torch.sin(nn_output[4])
    c = torch.cos(nn_output[4])
    nn_rot = torch.stack([torch.stack([c, -s]),
                       torch.stack([s, c])])
    nn_mu=torch.tensor([[nn_output[2].item()],[nn_output[3].item()]])
    nn_cov_sqr=torch.mm(torch.mm(nn_rot.double(),nn_mid.double()),torch.transpose(nn_rot.double(),0,1))
    nn_cov=torch.mm(nn_cov_sqr,nn_cov_sqr)

    label_mid = torch.tensor([[(label[0].item()), 0], [0, (label[1].item())]])
    s = torch.sin(label[4])
    c = torch.cos(label[4])
    label_rot = torch.stack([torch.stack([c, -s]),
                          torch.stack([s, c])])
    label_mu = torch.tensor([[label[2].item()], [label[3].item()]])
    label_cov_sqr = torch.mm(torch.mm(label_rot.double(), label_mid.double()), torch.t(label_rot.double()))
    label_cov=torch.mm(label_cov_sqr,label_cov_sqr)

    dist=torch.sum(torch.square(nn_mu-label_mu))+torch.trace(nn_cov+label_cov-2*sqrtm(torch.mm(torch.mm(nn_cov_sqr, label_cov),nn_cov_sqr)))
    if torch.isnan(dist):
        print("loss=nan")
        exit()
    else:
        return dist

def ws_dist_batch(nn_output, label):
    batch_size=nn_output.size(0)
    distances=torch.zeros(batch_size,1)
    for k in range(batch_size):
        nn_mid = torch.tensor([[(nn_output[k][0].item()), 0], [0, (nn_output[k][1].item())]])
        s = torch.sin(nn_output[k][4])
        c = torch.cos(nn_output[k][4])
        nn_rot = torch.stack([torch.stack([c, -s]),
                               torch.stack([s, c])])
        nn_mu = torch.tensor([[nn_output[k][2].item()], [nn_output[k][3].item()]])
        nn_cov_sqr = torch.mm(torch.mm(nn_rot.double(), nn_mid.double()), torch.transpose(nn_rot.double(), 0, 1))
        nn_cov = torch.mm(nn_cov_sqr, nn_cov_sqr)

        label_mid = torch.tensor([[(label[k][0].item()), 0], [0, (label[k][1].item())]])
        s = torch.sin(label[k][4])
        c = torch.cos(label[k][4])
        label_rot = torch.stack([torch.stack([c, -s]),
                                 torch.stack([s, c])])
        label_mu = torch.tensor([[label[k][2].item()], [label[k][3].item()]])
        label_cov_sqr = torch.mm(torch.mm(label_rot.double(), label_mid.double()), torch.t(label_rot.double()))
        label_cov = torch.mm(label_cov_sqr, label_cov_sqr)

        dist = torch.sum(torch.square(nn_mu - label_mu)) + torch.trace(
            nn_cov + label_cov - 2 * sqrtm(torch.mm(torch.mm(nn_cov_sqr, label_cov), nn_cov_sqr)))
        distances[k]=dist
        mean=torch.mean(distances)
    return mean
loss_fn = nn.MSELoss()

model = MyRNN(2, 16, 3, 5)
model.load_state_dict(torch.load('trained_model_rnn_32-5.pth'))
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


validate_data=torch.load('v_data_rnn.t')
#validate_data=validate_data.unflatten(1,(10,2))
validate_label=torch.load('v_label_rnn.t')
batch_size=1
val_loader=torch.utils.data.DataLoader(list(zip(validate_data, validate_label)),shuffle=False, batch_size=batch_size)
seq_length=torch.ones(1)*10
for validate_data, validate_label in val_loader:
    validate_data = validate_data.unflatten(1, (validate_data.size(1) // 2, 2))
    label_counter=0
    output=model(validate_data)
    output = output.squeeze(0)
    label = validate_label[0]
    geo_parameters = output.detach().numpy()
    ellipse=geo_parameters
    measurements = validate_data[0]
    x_measurement_wnoise = measurements[:, 0]
    y_measurement_wnoise = measurements[:, 1]
    plt.figure()
    plt.scatter(x_measurement_wnoise, y_measurement_wnoise, color='red', label='Sampled Points')
    A = (label[0]+1)*10
    B = (label[1]+1)*10
    H = label[2]*4
    K = label[3]*4
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

    A = (ellipse[0]+1) * 10
    B = (ellipse[1]+1) * 10
    H = ellipse[2] * 4
    K = ellipse[3] * 4
    tau = ellipse[4] * 3.14
    expanded_output = [A, B, H, K, tau]
    print("model output:", expanded_output)
    print("ws distance:", ws_dist(torch.tensor(expanded_output), torch.tensor(expanded_label)).item())
    print("MSE distance:",loss_fn(torch.tensor(expanded_output), torch.tensor(expanded_label)).item())
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


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sqrtm import sqrtm

class MyNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyNN, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.leakyrelu1 = nn.Sigmoid()
        nn.init.xavier_normal_(self.hidden1.weight)
        nn.init.zeros_(self.hidden1.bias)
        self.dropout1=nn.Dropout(0.1)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.hidden2.weight)
        nn.init.zeros_(self.hidden2.bias)
        self.leakyrelu2 = nn.Sigmoid()
        self.dropout2 = nn.Dropout(0.1)
        self.hidden3 = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.hidden3.weight)
        nn.init.zeros_(self.hidden3.bias)
        self.leakyrelu3 = nn.Sigmoid()
        self.dropout3 = nn.Dropout(0.1)
        self.hidden4 = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.hidden4.weight)
        nn.init.zeros_(self.hidden4.bias)
        self.leakyrelu4 = nn.Sigmoid()
        self.dropout4 = nn.Dropout(0.1)
        self.hidden5 = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.hidden5.weight)
        nn.init.zeros_(self.hidden5.bias)
        self.leakyrelu5 = nn.Sigmoid()
        self.dropout5 = nn.Dropout(0.1)
        self.hidden6 = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.hidden5.weight)
        nn.init.zeros_(self.hidden5.bias)
        self.leakyrelu6 = nn.Sigmoid()
        self.dropout6 = nn.Dropout(0.1)
        self.hidden7 = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.hidden5.weight)
        nn.init.zeros_(self.hidden5.bias)
        self.leakyrelu7 = nn.Sigmoid()
        self.dropout7 = nn.Dropout(0.1)
        self.hidden8 = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.hidden5.weight)
        nn.init.zeros_(self.hidden5.bias)
        self.leakyrelu8 = nn.Sigmoid()
        self.dropout8 = nn.Dropout(0.1)
        self.hidden9 = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.hidden5.weight)
        nn.init.zeros_(self.hidden5.bias)
        self.leakyrelu9 = nn.Sigmoid()
        self.dropout9 = nn.Dropout(0.1)
        self.hidden10 = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.hidden5.weight)
        nn.init.zeros_(self.hidden5.bias)
        self.leakyrelu10 = nn.Sigmoid()
        self.dropout10 = nn.Dropout(0.1)
        self.hidden11 = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.hidden5.weight)
        nn.init.zeros_(self.hidden5.bias)
        self.leakyrelu11 = nn.Sigmoid()
        self.dropout11 = nn.Dropout(0.1)
        self.hidden12 = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.hidden5.weight)
        nn.init.zeros_(self.hidden5.bias)
        self.leakyrelu12 = nn.Sigmoid()
        self.dropout12 = nn.Dropout(0.1)
        self.hidden13 = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.hidden5.weight)
        nn.init.zeros_(self.hidden5.bias)
        self.leakyrelu13 = nn.Sigmoid()
        self.dropout13 = nn.Dropout(0.1)
        self.hidden14 = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.hidden5.weight)
        nn.init.zeros_(self.hidden5.bias)
        self.leakyrelu14 = nn.Sigmoid()
        self.dropout14 = nn.Dropout(0.1)
        self.hidden15 = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.hidden5.weight)
        nn.init.zeros_(self.hidden5.bias)
        self.leakyrelu15 = nn.Sigmoid()
        self.dropout15 = nn.Dropout(0.1)
        self.hidden16 = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.hidden5.weight)
        nn.init.zeros_(self.hidden5.bias)
        self.leakyrelu16 = nn.Sigmoid()
        self.dropout16 = nn.Dropout(0.1)
        self.hidden17 = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.hidden5.weight)
        nn.init.zeros_(self.hidden5.bias)
        self.leakyrelu17 = nn.Sigmoid()
        self.dropout17 = nn.Dropout(0.1)
        self.hidden18 = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.hidden5.weight)
        nn.init.zeros_(self.hidden5.bias)
        self.leakyrelu18 = nn.Sigmoid()
        self.dropout18 = nn.Dropout(0.1)
        self.hidden19 = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.hidden5.weight)
        nn.init.zeros_(self.hidden5.bias)
        self.leakyrelu19 = nn.Sigmoid()
        self.dropout19 = nn.Dropout(0.1)
        self.hidden20 = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.hidden5.weight)
        nn.init.zeros_(self.hidden5.bias)
        self.leakyrelu20 = nn.Sigmoid()
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.bn=nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.leakyrelu1(x)
        res_1=x
        x = self.hidden2(self.bn(x))
        x = self.leakyrelu2(x)
        x = self.dropout1(x)
        res_2=x
        x=x+res_1
        x = self.hidden3(self.bn(x))
        x = self.leakyrelu3(x)
        x = self.dropout2(x)
        res_3=x
        x = self.hidden4(self.bn(x))
        x = self.leakyrelu4(x)
        x = self.dropout3(x)
        res_4 = x
        x=x+res_2
        x = self.hidden5(self.bn(x))
        x = self.leakyrelu5(x)
        x = self.dropout4(x)
        x=x+res_3
        x = self.hidden6(self.bn(x))
        x = self.leakyrelu6(x)
        x = self.dropout5(x)
        res_5 = x
        x = x+ res_4
        x = self.hidden7(self.bn(x))
        x = self.leakyrelu7(x)
        x = self.dropout6(x)
        res_6 = x
        x = x + res_5
        x = self.hidden8(self.bn(x))
        x = self.leakyrelu8(x)
        x = self.dropout7(x)
        res_7 = x
        x = x + res_6
        x = self.hidden9(self.bn(x))
        x = self.leakyrelu9(x)
        x = self.dropout8(x)
        res_8 = x
        x = x + res_7
        x = self.hidden10(self.bn(x))
        x = self.leakyrelu10(x)
        x = self.dropout9(x)
        res_9 = x
        x = x + res_8
        x = self.hidden11(self.bn(x))
        x = self.leakyrelu11(x)
        x = self.dropout10(x)
        res_10 = x
        x = x + res_9
        x = self.hidden12(self.bn(x))
        x = self.leakyrelu12(x)
        x = self.dropout11(x)
        res_11 = x
        x = x + res_10
        x = self.hidden13(self.bn(x))
        x = self.leakyrelu13(x)
        x = self.dropout12(x)
        res_12 = x
        x = x + res_11
        x = self.hidden14(self.bn(x))
        x = self.leakyrelu14(x)
        x = self.dropout13(x)
        res_13 = x
        x = x + res_12
        x = self.hidden15(self.bn(x))
        x = self.leakyrelu15(x)
        x = self.dropout14(x)
        res_14 = x
        x = x + res_13
        x = self.hidden16(self.bn(x))
        x = self.leakyrelu16(x)
        x = self.dropout15(x)
        res_15 = x
        x = x + res_14
        x = self.hidden17(self.bn(x))
        x = self.leakyrelu17(x)
        x = self.dropout16(x)
        res_16 = x
        x = x + res_15
        x = self.hidden18(self.bn(x))
        x = self.leakyrelu18(x)
        x = self.dropout17(x)
        res_17 = x
        x = x + res_16
        x = self.hidden19(self.bn(x))
        x = self.leakyrelu19(x)
        x = self.dropout18(x)
        res_18 = x
        x = x + res_17
        x = self.hidden20(self.bn(x))
        x = self.leakyrelu20(x)
        x = self.dropout19(x)
        x = self.output_layer(x)
        return x


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

model = MyNN(20, 32, 5)
model.load_state_dict(torch.load('trained_model_nn_32-5.pth'))
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


validate_data=torch.load('v_data_90D.t')
validate_label=torch.load('v_label_90D.t')
val_loader=torch.utils.data.DataLoader(list(zip(validate_data, validate_label)),shuffle=False, batch_size=2)
for validate_data, validate_label in val_loader:
    label_counter=0
    output = model(validate_data)
    output = output.squeeze(0)
    label = validate_label.squeeze(0)
    val_loss = ws_dist_batch(output, label)
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
        print("model output:",expanded_output)
        print("ws distance:", ws_dist(torch.tensor(expanded_output), torch.tensor(expanded_label)).item())
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


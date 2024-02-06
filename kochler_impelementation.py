import numpy as np
import matplotlib.pyplot as plt
import scipy
from vmm1 import intersection_points
from funcs import superellipse_sampler_1
import matplotlib.animation as animation
import time
start_time = time.time()


mat = scipy.io.loadmat('STurnScenario_Data.mat')
data = mat["data"]
validate_data = data[0]
validate_label = data[1]

I_repeats=10
dt=0.1
teta=0.5
dimensions=2
x_00=np.zeros([6,1])
P_00=np.diag([10,5,1])
v_00=4
X_00=100*np.identity(dimensions)
F_kinematic = np.array([[1,dt,((dt)**2)/2],
               [0,1,dt],
               [0,0,np.exp(-dt/teta)]])
Identity_matrix=np.identity(dimensions)
D=0.01*(1-np.exp(-2*dt/teta))*np.diag([0,0,1])
H=np.array([[1,0,0]])
sigma_x=0.5**2
delta_a=0.1
x_kmkm=x_00
P_kmkm=P_00
v_kmkm=v_00
X_kmkm=X_00
frames=[]
for k in range(200):
    
    frame=[]
    print(k)
    current_meas=validate_data[k]
    frame.append(current_meas)
    label = validate_label[k].T
    A_label = (label[0]) * 10
    B_label = (label[1]) * 10
    H_label = (label[2])*75
    K_label = (label[3])*75
    tau_label = label[4]*np.pi
    q_label = label[5]*8 + 2
    expanded_label = [A_label, B_label, H_label, K_label, tau_label,q_label]
    frame.append(expanded_label)
    
    start_angle=int(np.arctan(current_meas[0,1]/current_meas[0,0])*180/np.pi/delta_a)
    stop_angle=int(np.arctan(current_meas[-1,1]/current_meas[-1,0])*2*180/np.pi/delta_a)
    angles=range(start_angle-7,stop_angle+7)
    
    nk=np.size(current_meas)//2
    x_kkm=np.kron(F_kinematic, Identity_matrix)@x_kmkm
    P_kkm=F_kinematic @ P_kmkm@ np.transpose(F_kinematic) + D;
    v_kkm=np.exp(-dt/teta)*v_kmkm
    X_kkm=((np.exp(-dt/teta)*v_kmkm-dimensions-1) / (v_kmkm-dimensions-1.0))*X_kmkm
    zk=np.array([1/(nk)*current_meas.sum(axis=0)])
    Zk=0
    for numz in range(nk):
        diff=np.array(current_meas[numz]-zk)
        Zk += diff.T @ diff
    S_kkm= H @ P_kkm @ np.transpose(H)+sigma_x/nk
    W_kkm = P_kkm @ np.transpose(H) *(S_kkm**-1)
    x_kk=x_kkm + np.kron(W_kkm,Identity_matrix) @ (zk.T - np.kron(H,Identity_matrix)@x_kkm)
    P_kk=P_kkm-W_kkm @ S_kkm @ W_kkm.T
    N_kkm=(zk-np.kron(H,Identity_matrix)@x_kkm)@np.transpose((zk-np.kron(H,Identity_matrix)@x_kkm))*S_kkm**(-1)
    X_kk=X_kkm+N_kkm+Zk
    v_kk=v_kkm+nk
    
    # E_X_kk=(1/(v_kk-6))*X_kk
    # eigvalues,eigvectors=np.linalg.eig(E_X_kk)
    # y_k=np.array([x_kk[0][0] , x_kk[1][0] , 2*eigvalues[0]**(0.5), 2*eigvalues[1]**(0.5)])
    
    E_Z_k=(1/(nk))*Zk
    eigvalues,eigvectors=np.linalg.eig(E_Z_k)
    y_k=np.array([zk[0][0] , zk[0][1] , 2*eigvalues[0]**(0.5), 2*eigvalues[1]**(0.5)])
    if k==0:
        E_X_kk=(1/(v_kk-6))*X_kk
        eigvalues,eigvectors=np.linalg.eig(E_X_kk)
        u_ki=np.array([x_kk[0][0] , x_kk[1][0] , 2*eigvalues[0]**(0.5), 2*eigvalues[1]**(0.5)])
    else:
        u_ki=np.array([u_ki[0]+x_kk[2][0]*dt,u_ki[1]+x_kk[3][0]*dt, u_ki[2],u_ki[3]])
    delta_a=0.1
    tau=np.arctan2((x_kk[3][0]),(x_kk[2][0]))
    K_constant=0.1
    error_im=np.array([np.inf,np.inf])
    mag_error_im=np.sqrt(error_im.dot(error_im))
    for i in range(I_repeats):
        virtual_measurements=np.zeros(shape=(len(angles),2))
        b=0
        ellipse_params=np.array([u_ki[0],u_ki[1],u_ki[2],u_ki[3],tau])
        for a in angles:
            intersection_point=intersection_points(ellipse_params,a*delta_a)
            if not len(intersection_point):
                virtual_measurements=np.delete(virtual_measurements,b,0)
            else:
                virtual_measurements[b]=intersection_point
                b=b+1
        vnk=virtual_measurements.size//2
        y_mean_ki=np.array([1/(vnk)*virtual_measurements.sum(axis=0)])
        Y_tilda_ki=0
        for numy in range(vnk):
            diff=np.array(virtual_measurements[numy]-y_mean_ki)
            Y_tilda_ki += diff.T@diff
        eigvalues_y,eigvectors_y=np.linalg.eig(Y_tilda_ki/vnk)
        y_tilda_ki=np.array([y_mean_ki[0][0],y_mean_ki[0][1],2*eigvalues_y[0]**(0.5),2*eigvalues_y[1]**(0.5)])
        error_i=(y_k-y_tilda_ki)
        mag_error_i=np.sqrt(error_i.dot(error_i))
        u_kip=u_ki+K_constant*(y_k-y_tilda_ki)
        if mag_error_i > mag_error_im:
            u_kip=u_ki
            K_constant=K_constant/2
        mag_error_im=mag_error_i
        u_ki=u_kip
    step_output=np.array([u_ki[0],u_ki[1],u_ki[2],u_ki[3],tau])
    frame.append(step_output)
    frame.append(virtual_measurements)
    frames.append(frame)
    x_kmkm=x_kk
    X_kmkm=X_kk
    P_kmkm=P_kk
    v_kmkm=v_kk
        
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
    virtual_meas=frame[3]
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
    
    A = expanded_output[2]
    B = expanded_output[3]
    H = expanded_output[0]
    K = expanded_output[1]
    tau = expanded_output[4]
    q = 2
    x_ellipse , y_ellipse = superellipse_sampler_1(A, B, H, K, tau, q, angles)
    ax.plot(x_ellipse, y_ellipse, label='Output Ellipse', color="green")
    
    x_measurement_wnoise = virtual_meas[:, 0]
    y_measurement_wnoise = virtual_meas[:, 1]
    ax.scatter(x_measurement_wnoise, y_measurement_wnoise, color='cyan', label='VirtualMeasurements')
    
    
    plt.title('Output Ellipse')
    ax.legend()
    ax.set_xlim((0,75));ax.set_ylim((0,75))
    plt.grid(True)


anim = animation.FuncAnimation(fig, update, frames=frames, repeat=False)
anim.save("KochImplementation.mp4",fps=1)
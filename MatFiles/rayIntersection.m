num_ellipse=5000;
num_samples = 2;
data = cell(3,(num_ellipse*num_samples));
% Scenarios= S_turn , Random
scenario="Random";
Scenario_data=zeros(5,num_ellipse);
% CV-T S turn scenario
if scenario == "S_turn"
A=6;
B=3;
V=5;
H=20;
K=20;
teta=0;
tau=0;
dt=0.1;
    for i = 1:num_ellipse
        disp(i)
        if i == num_ellipse/4
            teta=pi/(num_ellipse/4*dt);
        elseif i == num_ellipse/2
            teta=0;
        elseif i == num_ellipse*3/4
            teta=-pi/(num_ellipse/4*dt);
        end
        disp(teta)
        tau=tau+dt*teta;
        H=H+V*cos(tau)*dt;
        K=K+V*sin(tau)*dt;
        Scenario_data(:,i) = [A,B,H,K,tau];
    end
end

%Random Scenario
if scenario == "Random"
    for i = 1:num_ellipse
        B = rand*5;
        A = B+rand*5;
        H = rand*55+10;
        K = rand*55+10;
        tau = rand*pi;
        q = 2;
        Scenario_data(:,i) = [A,B,H,K,tau];
    end
end
for i = 0:num_ellipse-1
    map = binaryOccupancyMap(75,75,10);
    B = Scenario_data(2,i+1);
    A = Scenario_data(1,i+1);
    H = Scenario_data(3,i+1);
    K = Scenario_data(4,i+1);
    tau = Scenario_data(5,i+1);
    q = 2;
    label=[A/10,B/10,(H/75),(K/75),(tau/pi),(q-2)/8];
    angles = linspace(0, 2*pi, 250);
    q_dists = (abs(cos(angles + tau)).^q + abs(sin(angles + tau)).^q).^(-1/q);
    x = q_dists .* cos(angles + tau) .* A .* cos(tau) - q_dists .* sin(angles + tau) .* B .* sin(tau) + H;
    y = q_dists .* cos(angles + tau) .* A .* sin(tau) + q_dists .* sin(angles + tau) .* B .* cos(tau) + K;
    obstacles = vertcat(x,y).';
    setOccupancy(map,obstacles,ones(length(obstacles),1))
    inflate(map,0.05)
    maxrange = 200;
    angles = linspace(0,pi/2,250);
    vehiclePose = [0,0,0];
    intsectionPts = rayIntersection(map,vehiclePose,angles,maxrange);

    for j = 1:num_samples
        prior=[A/10+randn*0.02,B/10+randn*0.02,(H/75)+randn*0.1,(K/75)+randn*0.1,(tau/pi)+randn*0.2,(q-2)/8+randn*0.03];
        range_meas = sqrt(intsectionPts(:, 1).^2 + intsectionPts(:, 2).^2);
        angle_meas = atan2(intsectionPts(:, 2), intsectionPts(:, 1));
        range_noise = 0.05 * randn(size(intsectionPts, 1), 1);
        angle_noise = deg2rad(0.5) * randn(size(intsectionPts, 1), 1);
        noisy_range = range_meas + range_noise;
        noisy_angle = angle_meas + angle_noise;
        
        noisy_meas_x = noisy_range .* cos(noisy_angle);
        noisy_meas_y = noisy_range .* sin(noisy_angle);
        noisy_meas_x = noisy_meas_x(~isnan(noisy_meas_x));
        noisy_meas_y = noisy_meas_y(~isnan(noisy_meas_y));
        noisy_measurement = [noisy_meas_x, noisy_meas_y];


        % show(map)
        % hold on
        % grid on
        % plot(intsectionPts(:,1),intsectionPts(:,2),'*r') % Intersection points
        % plot(vehiclePose(1),vehiclePose(2),'ob') % Vehicle pose
        % pause(0.1)

        data{1,num_samples*i+j}=noisy_measurement;
        data{2,num_samples*i+j}=label;
        data{3,num_samples*i+j}=prior;
    end
end
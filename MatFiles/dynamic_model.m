num_ellipse=1;
sim_length=50;
num_samples = 1;
data = cell(3,(num_ellipse*num_samples*sim_length));
for i = 0:num_ellipse-1
    B = rand*4+1;
    A = B + rand*4+1;
    tau = pi*(rand);
    q = rand*8+2;
    dt=1;
    X=[10,1,10,1].';
    A_mat=[1,dt,0,0;
        0,1,0,0;
        0,0,1,dt;
        0,0,0,1];
    for o=1:sim_length
        model_noise=([0,1,0,1]*randn*0.3).';
        X=A_mat*X+model_noise;
        H=X(1);
        K=X(3);
        label=[A/10,B/10,(H/50),(K/50),(tau/pi),(q-2)/8];
        angles = linspace(0, 2*pi, 250);
        q_dists = (abs(cos(angles + tau)).^q + abs(sin(angles + tau)).^q).^(-1/q);
        x = q_dists .* cos(angles + tau) .* A .* cos(tau) - q_dists .* sin(angles + tau) .* B .* sin(tau) + H;
        y = q_dists .* cos(angles + tau) .* A .* sin(tau) + q_dists .* sin(angles + tau) .* B .* cos(tau) + K;
        obstacles = vertcat(x,y).';
        map = binaryOccupancyMap(50,50,10);
        setOccupancy(map,obstacles,ones(length(obstacles),1))
        inflate(map,0.05)
        maxrange = 200;
        angles = linspace(0,pi/2,250);
        vehiclePose = [0,0,0];
        intsectionPts = rayIntersection(map,vehiclePose,angles,maxrange);
    
        for j = 1:num_samples
            prior=[A/10+randn*0.02,B/10+randn*0.02,(H/50)+randn*0.1,(K/50)+randn*0.1,(tau/pi)+randn*0.2,(q-2)/8+randn*0.03];
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
            data{1,sim_length*num_samples*i+o-1+j}=noisy_measurement;
            data{2,sim_length*num_samples*i+o-1+j}=label;
            data{3,sim_length*num_samples*i+o-1+j}=prior;
        end
    end

    
end
clear;

load Data.mat;

% load T_Spatial;

% load('./Data/SST-Sattelite-2016.mat');

% Day = 78;

% T = Data.SST(10:32,1:25,Day);
% 
% N = 99;
% 
% x_lf = zeros(N, 1);
% 
% y_lf = zeros(N, 1);
% 
% T_lf = zeros(N, 1);
% 
% i = 1;
% 
% for x = 10:32
%     for y = 1:25
%         if (~isnan(T(x - 10 + 1, y)))
%             x_lf(i, 1) = Data.X(x, y);
%             y_lf(i, 1) = Data.Y(x, y);
%             T_lf(i, 1) = T(x - 10 + 1, y);
%             i = i +1;
%         end
%     end
% end

x_lf = LF.X(:, 1);

y_lf = LF.X(:, 2);

% hold on;

% plot3(x_lf, y_lf, T_lf, 'r*');

% load u_pred_l;
% 
% N = 81;
% 
% x_test = Pred(1).X(:, 1);
% 
% y_test = Pred(1).X(:, 2);
% 
% x = reshape(x_test, N, N);
% 
% y = reshape(y_test, N, N);

% mesh(x, y, u_pred_l);

T_lf = LF.Y;

x_m = HF.X(:, 1);

y_m= HF.X(:, 2);

T_m = HF.Y;

n_train = [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14];

n_test = [2, 10];

x_hf = x_m(n_train);

y_hf = y_m(n_train);

T_hf = T_m(n_train);

x_vld = x_m(n_test);

y_vld = y_m(n_test);

T_vld = T_m(n_test);

% load Station_info.mat;
% 
% n_vld = [5, 11];
% 
% y_vld = Station.lat(n_vld);
% 
% x_vld = Station.lon(n_vld);
% 
% T_vld = Pred(2).Y;

filename = 'T_Spatial';

save(filename, 'x_lf', 'y_lf', 'T_lf', 'x_hf', 'y_hf', 'T_hf', 'x_vld', 'y_vld', 'T_vld');
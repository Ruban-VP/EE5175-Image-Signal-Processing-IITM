%{
************************************************************************************
EE5175: Image signal processing - Lab 09 MATLAB code
Author: V. Ruban Vishnu Pandian (EE19B138)
Date: 01/04/2023
Note: Ensure all the source images are present in the present working 
directory and the Image processing toolbox is installed.
************************************************************************************
%}

%% Lab 9: DCT, Walsh-Hadamard transform and SVD

clear;
close all;

%% Question - 1

N = 8;
rho = 0.91;           % N and rho values 

inds = 0:N-1;
row = rho.^inds;
R = toeplitz(row);    % Covariance matrix for given Markov-1 process

[R_eig,D_eig] = eig(R); % Eigenvalue decomposition of R

% For loop to create the DCT matrix as asked
DCT_matrix = zeros(N,N);
DCT_matrix(1,:) = ones(1,N)/sqrt(2);
for k=1:N-1
    for n=0:N-1
        DCT_matrix(k+1,n+1) = cos(pi*(2*n+1)*k/(2*N));
    end
end
DCT_matrix = DCT_matrix*sqrt(2/N);

H = [1 1; 1 -1]/sqrt(2);         % Walsh-Hadamard transform matrix
WHT_matrix = kron(kron(H,H),H);  % of order 8 is created 

% Unitary transformations applied on the covariance matrix
R_DCT = DCT_matrix*R*DCT_matrix';
R_WHT = WHT_matrix*R*WHT_matrix';

EPE_DCT = zeros(1,N);   % Arrays to store the energy packing efficiency 
EPE_WHT = zeros(1,N);   % (EPE) values for different M values   
val = 0;
val_DCT = 0;
val_WHT = 0;
val_abs = 0;
val_DCT_abs = 0;
val_WHT_abs = 0;

% For loop to compute EPE values for different M values
for i=1:N
	val = val+R(i,i);
    val_abs = val_abs+abs(R(i,i));
	val_DCT = val_DCT+R_DCT(i,i);
	val_WHT = val_WHT+R_WHT(i,i);
    val_DCT_abs = val_DCT_abs+abs(R_DCT(i,i));
	val_WHT_abs = val_WHT_abs+abs(R_WHT(i,i));

	EPE_DCT(i) = val_DCT;
	EPE_WHT(i) = val_WHT;
end

EPE_DCT = EPE_DCT/val_DCT;
EPE_WHT = EPE_WHT/val_WHT;

% Data decorrelation efficiency (DDE) computed for both the 
% transforms
DDE_DCT = 1-((sum(sum(abs(R_DCT)))-val_DCT_abs)/(sum(sum(abs(R)))-val_abs));
DDE_WHT = 1-((sum(sum(abs(R_WHT)))-val_WHT_abs)/(sum(sum(abs(R)))-val_abs));

% Plot of EPE vs M
figure(1)
plot(1:N,EPE_DCT,'LineWidth',2)
hold on
plot(1:N,EPE_WHT,'LineWidth',2)
xlabel("M values")
ylabel("EPE")
title("EPE vs M plot")
legend('DCT','WHT','Location','Southeast')

%% Question - 2

beta_sq = (1-(rho*rho))/(1+(rho*rho));
alpha = rho/(1+(rho*rho));    % Formulae for alpha and beta squared

Q_est = beta_sq*(R\eye(N));   % Inverse of R scaled with beta squared
Q_act = zeros(N,N);           % Tri-diagonal matrix with alpha as given

Q_act(1,1) = 1-alpha;
Q_act(1,2) = -alpha;
Q_act(N,N-1) = -alpha;
Q_act(N,N) = 1-alpha;

for i=2:N-1
	Q_act(i,i-1) = -alpha;
	Q_act(i,i) = 1;
	Q_act(i,i+1) = -alpha;    % Tri-diagonal Q matrix created
end

% DCT transform applied on both the Q matrices as asked
Q_est_diag = DCT_matrix*Q_est*DCT_matrix';
Q_act_diag = DCT_matrix*Q_act*DCT_matrix';

%% Question - 3

load('imageFile.mat')         % Input image loaded

g_gt = g*g';
gt_g = g'*g;              

[A,D1] = eig(g_gt,'vector');  % A,B matrices obtained using 
[B,D2] = eig(gt_g,'vector');  % eigen decomposition

D = sort(D1,'descend');       % Eigenvalues sorted in descending order
thr = 1e-10;
P = N;

% For loop to re-assign zero eigenvalues and compute rank
for i=1:N
    if D(i)<thr
        D(i)=0;                 
        P = P-1;              
    end
end

D = D(1:P);
sing = sqrt(D);
A_vecs = zeros(N,P);
B_vecs = zeros(N,P);

% For loop to obtain column vectors of A and B arranged in 
% descending order of singular values
for i=1:P
    [~,ind1] = min(abs(D1-D(i)));
    [~,ind2] = min(abs(D2-D(i)));
    a_vec = A(:,ind1);
    b_vec = B(:,ind2);

    vec1 = g*b_vec;
    vec2 = sing(i)*a_vec;
    sign = mean(vec1./vec2);

    A_vecs(:,i) = a_vec;
    B_vecs(:,i) = sign*b_vec;
end

% Image reconstructed using A, B and singular value matrix as asked
g_rec = A_vecs*diag(sing)*B_vecs';

%% Question - 4

error_obs = zeros(1,P+1);   % Array to store the squared errors
error_calc = zeros(1,P+1);  % Array to store the sum of square of singular values 
sum_eig = 0;

g_im = mat2gray(g);
imwrite(g_im,'g.png');      % Input image stored as .png file

for i=1:P
    % Image reconstructed by eliminating one eigenvalue at once
    g_est = A_vecs(:,i+1:P)*diag(sing(i+1:P))*B_vecs(:,i+1:P)';
    sum_eig = sum_eig + (sing(i)^2);  % Sum of square of singular values

    % Low rank image stored for each iteration
    g_est_im = mat2gray(g_est);
    imwrite(g_est_im,strcat('g_est_',num2str(i),'.png'));

    % Error computed directly and using sum of square of singular values
    error_obs(i+1) = norm(g-g_est,"fro")^2;
    error_calc(i+1) = sum_eig;
end

% Plot of error against k (First k singular values are removed)
figure(2)
plot(0:P,error_obs,'LineWidth',2)
xlabel("k values")
ylabel("Residual error")
title("Error vs k plot")
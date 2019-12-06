
clc

%%
% Approximated differentiation


% Sampling frequency
Ts = 0.001 % s

% Cut-off frequency
% fc = 100 % Hz
fc = 200 % Hz

%
S = tf([1 0], [1]);
%
LPF1 = tf([1],[1/(2*pi*fc) 1]);
% LPF2 = LPF1*LPF1; % 2nd-order
% LPF2 = LPF1*LPF1*LPF1; % 3rd-order
% LPF2 = LPF1*LPF1*LPF1*LPF1; % 4th-order
LPF2 = LPF1*LPF1*LPF1*LPF1*LPF1*LPF1; % 6th-order

% Approximated differentiation
S_apx = S*LPF2

% Discrete version
S_apx_d = c2d(S_apx, Ts,'tustin')

% Get parameters
[num,den] = tfdata(S_apx_d,'v')

%%
figure(1)
bode(S,S_apx,S_apx_d)

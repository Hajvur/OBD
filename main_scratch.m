close all;
clear all;
clc
% dane do sieci

x1 = [0 0 1 1];
x2 = [0 1 0 1];
y = [0 1 1 0];

x = [x1; x2];

% zalozenia sieci
H1 = 3;
H2 = 2;
R1 = 2;
R2 = 1;
% 2 neurony wejsciowe
% 1 wyjsciowy
% 2 warstwy ukryte, 3 i 2 neurony
w1 = randn(H1, R1) * 0, 1;
w2 = randn(H2, H1) * 0, 1;
w3 = randn(R1, H2) * 0, 1;
b1 = randn(H1, 1) * 0, 1;
b2 = randn(H2, 1) * 0, 1;
b3 = randn(R1, 1) * 0, 1;

% funkcje pomocnicze

function [y] = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end

function [y] = Re_LU(x)
    y = max(0, x);
end

function [] = forwardpass()
end

function [] = backpropagation()
end

function [] = compute_gradient()

end

function [] = compute_jacobian()

end

function [] = compute_hessian()

end

function [] = OBD()
end

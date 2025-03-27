close all;
clear all;
clc
% dane do sieci

x1 = [0 0 1 1];
x2 = [0 1 0 1];
y = [0 1 1 0];

x = [x1; x2];
% zalozenia sieci
epochs = 5000;
learning_rate = 0.5;
tolerance = 0.0001;
lambda_backpropagation = 0.001;
lambda_lm = 0.001;
% liczba neuronow w warstwie wejsciowej, ukrytej i wyjsciowej
H1 = 3;
H2 = 2;
R1 = 2;
R2 = 1;
% 2 neurony wejsciowe
% 1 wyjsciowy
% 2 warstwy ukryte, 3 i 2 neurony
%zastosowalem scaling Xaviera
rng(0)
w1 = randn(H1, R1) * sqrt(2 / (R1 + H1));
w2 = randn(H2, H1) * sqrt(2 / (H1 + H2));
w3 = randn(R2, H2) * sqrt(2 / (H2 + R2));
b1 = zeros(H1, 1);
b2 = zeros(H2, 1);
b3 = zeros(R2, 1);

% funkcje pomocnicze

function [y] = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end

function [y] = Re_LU(x)
    y = max(0, x);
end

function [mse_ex] = mse(error)
    mse_ex = mean(error .^ 2);
end

function [y_predict] = forwardpass_backpropagation(w1, w2, w3, b1, b2, b3, x, y, epochs, tolerance, learning_rate, lambda)
    %forward pass
    for epoch = 1:epochs
        z1 = w1 * x + b1;
        a1 = sigmoid(z1);
        z2 = w2 * a1 + b2;
        a2 = sigmoid(z2);
        z3 = w3 * a2 + b3;
        y_predict = sigmoid(z3);
        % obliczanie bledu
        error = y_predict - y;
        %disp(error)
        mse_ex = mse(error);
        %disp("mse ex");
        %disp(mse_ex)

        if mse_ex < tolerance
            break
        end

        %backpropagation (póki co tylko testy na pochodnych sigmoid)
        % pochodna sigmoid sigmoid(z3) * (1-sigmoid(z3)) albo podstawiamy od razu y_predict
        [gw1, gw2, gw3, gb1, gb2, gb3] = compute_gradient(error, w2, w3, a1, a2, x, y_predict);

        w1 = w1 - learning_rate * (gw1 + lambda * w1);
        w2 = w2 - learning_rate * (gw2 + lambda * w2);
        w3 = w3 - learning_rate * (gw3 + lambda * w3);

        b1 = b1 - learning_rate * gb1;
        b2 = b2 - learning_rate * gb2;
        b3 = b3 - learning_rate * gb3;
        %disp("w3")
        %disp(w3);
        %disp("w2")
        %disp(w2);
        %disp("w1")
        %disp(w1);
    end

end

function [y_predict] = forwardpass_lm_obd(w1, w2, w3, b1, b2, b3, x, y, epochs, tolerance, lambda)
    %forward pass
    for epoch = 1:epochs
        z1 = w1 * x + b1;
        a1 = sigmoid(z1);
        z2 = w2 * a1 + b2;
        a2 = sigmoid(z2);
        z3 = w3 * a2 + b3;
        y_predict = sigmoid(z3);
        % obliczanie bledu
        error = y_predict - y;
        %disp(error)
        mse_ex = mse(error);
        disp("mse ex");
        disp(mse_ex)

        if mse_ex < tolerance
            break
        end

        J_w1 = zeros(size(x, 2), numel(w1));
        J_b1 = zeros(size(x, 2), numel(b1));
        J_w2 = zeros(size(x, 2), numel(w2));
        J_b2 = zeros(size(x, 2), numel(b2));
        J_w3 = zeros(size(x, 2), numel(w3));
        J_b3 = zeros(size(x, 2), numel(b3));

        for i = 1:size(x, 2)
            dw3 = error(:, i) .* y_predict(:, i) .* (1 - y_predict(:, i));
            dw2 = (w3' * dw3) .* a2(:, i) .* (1 - a2(:, i));
            dw1 = (w2' * dw2) .* a1(:, i) .* (1 - a1(:, i));

            J_w3(i, :) = reshape(dw3 * a2(:, i)', 1, []);
            J_b3(i, :) = dw3;

            J_w2(i, :) = reshape(dw2 * a1(:, i)', 1, []);
            J_b2(i, :) = dw2;

            J_w1(i, :) = reshape(dw1 * x(:, i)', 1, []);
            J_b1(i, :) = dw1;
        end

        %disp("j1")
        %disp(J_w1)
        disp("hessian diag")
        disp(diag(J_w1' * J_w1))
        disp ("hessia n di")
        disp(sum(J_w1 .^ 2, 1)')

        w1 = uptade_weights(w1, J_w1, error, lambda);
        w2 = uptade_weights(w2, J_w2, error, lambda);
        w3 = uptade_weights(w3, J_w3, error, lambda);
        b1 = uptade_weights(b1, J_b1, error, lambda);
        b2 = uptade_weights(b2, J_b2, error, lambda);
        b3 = uptade_weights(b3, J_b3, error, lambda);
        %obd (pozniej bedzie to zaimplementowane w funkcji)
        if mod(epoch, 100) == 0
            %mamy juz pochodne czastkowe w jakobianach
            % hessian to jest J' * J
            % jeżeli diagonalne elementy hessiana sa pochodnymi dla danego parametru
            % to sum daje to samo co diag co wyżej w za pomoca disp jest
            % powtwierdzone
            H_diag_w1 = sum(J_w1 .^ 2, 1)';
            H_diag_w2 = sum(J_w2 .^ 2, 1)';
            H_diag_w3 = sum(J_w3 .^ 2, 1)';

            saliency_w1 = 0.5 * (H_diag_w1 .* w1(:) .^ 2);
            saliency_w2 = 0.5 * (H_diag_w2 .* w2(:) .^ 2);
            saliency_w3 = 0.5 * (H_diag_w3 .* w3(:) .^ 2);

            w1 = prune(w1, saliency_w1, 0.1);
            w2 = prune(w2, saliency_w2, 0.1);
            w3 = prune(w3, saliency_w3, 0.1);
            %disp("waga po usunieciu")

        end

    end

end

function [gw1, gw2, gw3, gb1, gb2, gb3] = compute_gradient(error, w2, w3, a1, a2, x, y_predict)

    dw3 = error .* y_predict .* (1 - y_predict);
    dw2 = (w3' * dw3) .* a2 .* (1 - a2);
    dw1 = (w2' * dw2) .* a1 .* (1 - a1);

    gw3 = dw3 * a2';
    gw2 = dw2 * a1';
    gw1 = dw1 * x';

    gb3 = sum(dw3, 2);
    gb2 = sum(dw2, 2);
    gb1 = sum(dw1, 2);

end

function [waga] = uptade_weights(waga, jakobian, error, lambda)
    H = jakobian' * jakobian + lambda * eye(size(jakobian, 2));
    %disp("waga size:")
    %disp(size(waga))
    %disp("reszta size")
    %disp(size(H \ (jakobian' * error(:))))
    delta =- (H \ (jakobian' * error(:)));
    waga = waga + reshape(delta, size(waga));
end

function [waga] = prune(wagi, saliency, procent_usuniecia)
    [elements, idx] = sort(saliency);
    pruning = floor(procent_usuniecia * numel(wagi));
    waga(idx(1:pruning)) = 0;

end

y_predict = forwardpass_backpropagation(w1, w2, w3, b1, b2, b3, x, y, epochs, tolerance, learning_rate, lambda_backpropagation);
disp('Predykcja back:')
disp(y_predict)

y_predict_lm = forwardpass_lm_obd(w1, w2, w3, b1, b2, b3, x, y, 200, tolerance, lambda_lm);
disp('Predykcja LM:')
disp(y_predict_lm)

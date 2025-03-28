clc
clear all
close all
warning off

function [y] = tansig(x)
    y = tanh(x);
end

function [y] = purelin(x)
    y = x;
end

function [mse_ex] = mse(error)
    mse_ex = mean(error .^ 2);
end

load train_data.mat
x = train_data';
xt = test_data';

epochs = 2;
learning_rate = 0.01;
%wagi i biasy

load init_weights.mat
w1 = wh1';
w2 = wh2';
w3 = wo';
b1 = b1';
b2 = b2';
b3 = bo';

%forward pass
tic;

for epoch = 1:epochs
    z1 = w1 * x + b1;
    a1 = tansig(z1);
    z2 = w2 * a1 + b2;
    a2 = tansig(z2);
    z3 = w3 * a2 + b3;
    y_predict = purelin(z3);

    % obliczanie bledu
    error = y_predict - y;
    mse_error = mse(error);
    disp(['MSE treningowe (epoka ', num2str(epoch), '): ', num2str(mse_error)]);

    %if mse_ex < tolerance
    %    break
    %end

    dw3 = error;
    dw2 = (w3' * dw3) .* (1 - a2 .^ 2);
    dw1 = (w2' * dw2) .* (1 - a1 .^ 2);

    gw3 = (dw3 * a2') / size(x, 2);
    gw2 = (dw2 * a1') / size(x, 2);
    gw1 = (dw1 * x') / size(x, 2);

    gb3 = sum(dw3, 2) / size(x, 2);
    gb2 = sum(dw2, 2) / size(x, 2);
    gb1 = sum(dw1, 2) / size(x, 2);

    w1 = w1 - learning_rate * gw1;
    w2 = w2 - learning_rate * gw2;
    w3 = w3 - learning_rate * gw3;

    b1 = b1 - learning_rate * gb1;
    b2 = b2 - learning_rate * gb2;
    b3 = b3 - learning_rate * gb3;
end

tr_time = toc;

%dane testowe
z1_test = w1 * xt + b1;
a1_test = tansig(z1_test);
z2_test = w2 * a1_test + b2;
a2_test = tansig(z2_test);
z3_test = w3 * a2_test + b3;
ynn = purelin(z3_test);

% nie było y_test w zestawie danych, takze poki co uzywam y
test_error = ynn - y;
mse_test_error = mse(test_error);
disp(['MSE testowe: ', num2str(mse_test_error)]);

figure(2);
plot(ynn, 'g', 'LineWidth', 1.5);
hold on;
plot(y, 'b', 'LineWidth', 1.5);
grid on;
legend('Przewidziane', 'Rzeczywiste', 'Location', 'best');
xlabel('Numer próbki');
ylabel('Wartość');
title('Porównanie wyników sieci');
box on;

disp('-------------------------------');
disp(['Czas treningu: ', num2str(tr_time), ' sekund']);
disp(['Błąd MSE (testowe): ', num2str(mse_test_error)]);
disp('-------------------------------');
wh1

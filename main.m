close all;
clear all;
clc
% trenowanie sieci

x1 = [0 0 1 1];
x2 = [0 1 0 1];
y = [0 1 1 0];

x = [x1; x2];

siec_newff = feedforwardnet([20,10])
siec_newff.trainParam.epochs = 200;
siec_newff.trainParam.goal = 0;
siec_newff = train(siec_newff, x, y);

y_pred = sim(siec_newff,x);
mse_org = mean((y-y_pred).^2);
disp("Mse org: "+mse_org);

num_weight_org = length(getwb(siec_newff));
disp("liczba wag przed: "+ num_weight_org);

% czesc algorytmu obd
% 1 wyznaczenie funcji błedu
% 2 obliczenie gradientu
% 3 obliczenie drugich pochodnych diagonalnych macierzy hessiego
% 4 wyznaczenie "istotnosci" wag (damy 50 percentyl z istotnosci)
% 5 usuniecie wag ponizej poziomu istotnosci
% i to iteracyjnie

J_e = defaultderiv('de_dwb',siec_newff,x,y);
hessian = J_e.' * J_e; % dla testow
hess_diag = diag(J_e.' * J_e);
%disp("oryginalne");
%disp(getwb(siec_newff));



for i=1:2
    weights= getwb(siec_newff);
    saliency = 0.5 *(hess_diag(i).* (weights.^2));
    sorted_saliency= sort(saliency);
    mid_saliency = sorted_saliency(ceil(end/2),:);
    weights(saliency<mid_saliency)=0;
    %disp("po iteracji petli");
    %disp(weights);
    siec_newff= setwb(siec_newff,weights);
 end



y_pred_obd = sim(siec_newff,x);
mse_after = mean((y-y_pred).^2);
disp("Mse obd: "+mse_after);

num_weight_obd = sum(getwb(siec_newff) ~=0);
disp("liczba wag po: "+ num_weight_obd);


figure;
subplot(1,2,1);
bar(y_pred);
title("Predykcje przed OBD");

subplot(1,2,2);
bar(y_pred_obd);
title("Predykcje po OBD");




    
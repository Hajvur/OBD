clc
clear all
close all
warning off

%data from file
clear all
load train_data
x=train_data';
xt=test_data';
%%%%%%%%%%%%%%%%%%%%%%%%netwhork parameters
%%%topology
zakres=[-1 1];
liczba_n_h1=17;
liczba_n_h2=18;
liczba_n_o=1;

NN_model=newff([zakres;zakres;zakres],[liczba_n_h1 liczba_n_h2 liczba_n_o],{'tansig','tansig','purelin'},'traingd');

NN_model.trainParam.epochs=2;
NN_model.trainParam.lr=0.01;

%init from file
load init_weights.mat
NN_model.IW{1,1}=wh1';
NN_model.LW{2,1}=wh2';
NN_model.LW{3,2}=wo';
NN_model.b{1,1}=b1';
NN_model.b{2,1}=b2';
NN_model.b{3,1}=bo';

NN_model.performParam.normalization='none';
NN_model.performParam.regularization=0;
%%%%%%%%%%%%%%%%%%%%%%%%training
tic;
NN_model_t=train(NN_model,x,y);
training_time=toc;
ynn=sim(NN_model_t,xt);

total_error=mse(ynn,y);

figure(2)
plot(ynn,'g');grid;hold;plot(y);legend('nn','real');xlabel('number of sample');ylabel('values');box on
disp('-------------------------------')
disp('training time (seconds):')
disp(training_time)
disp('total error value (mse):')
disp(total_error)
disp('-------------------------------')

%%%%%%%%%%%%%%%%%%%

bo=NN_model_t.b{3,1};
b2=NN_model_t.b{2,1};
b1=NN_model_t.b{1,1};
wo=NN_model_t.LW{3,2};
wh2=NN_model_t.LW{2,1};
wh1=NN_model_t.IW{1,1};

clc
%wh1

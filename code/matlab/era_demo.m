clear all
close all
load testSys

%% Dsiplay full system info
numInputs, numOutputs
size(sysFull.A)
size(sysFull.B)
size(sysFull.C)
size(sysFull.D)

%% Plot impulse response of full system
figure
impulse(sysFull,0:1:(r*5)+1);
grid;
set(gcf,'position',[100,700,800,400]);
saveas(gcf,"../../plots/era_sys_imp.png")

%% Obtain impulse response of full system
[yFull,t] = impulse(sysFull,0:1:(r*5)+1);  
YY = permute(yFull,[2 3 1]); % Reorder to be size p x q x m 
                             % (default is m x p x q)
%% Compute ERA from impulse response
mco = floor((length(yFull)-1)/2)
[Ar,Br,Cr,Dr,HSVs] = ERA(YY,mco,mco,numInputs,numOutputs,r);
sysERA = ss(Ar,Br,Cr,Dr,-1);

%% Plot impulse responses for both models
figure
impulse(sysFull, sysERA, 50);
grid
legend('Full model, n=100',['ERA, r=',num2str(r)])
set(gcf,'position',[100,400,800,400]);
set(gcf,'PaperPositionMode','auto')
saveas(gcf,'../../plots/era_sys2_imp.png')


%% Compare Bode plots
figure
bode(sysFull, sysERA);
set(gcf,'position',[100,100,800,400]);
set(gcf,'PaperPositionMode','auto')
saveas(gcf,'../../plots/era_sys2_bode.png')


%% Gramians
Wc = gram(sysERA,'c') % Controllability Gramian
Wo = gram(sysERA,'o') % Observability Gramian

figure
plot(diag(Wc))
hold on
plot(diag(Wo))
legend(["W_c", "W_o"])
grid
title("Diagonals of Gramians - reduced system model")
saveas(gcf,'../../plots/era_sys2_gramdiag.png')
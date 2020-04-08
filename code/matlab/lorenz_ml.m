% Simulate Lorenz system
dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; r=28;
Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

figure(1)
input=[]; output=[];
for j=1:100 % training trajectories
    x0=30*(rand(3,1)-0.5);
    [t,y] = ode45(Lorenz,t,x0);
    input=[input; y(1:end-1,:)];
    output=[output; y(2:end,:)];
    plot3(y(:,1),y(:,2),y(:,3)), hold on
    plot3(x0(1),x0(2),x0(3),'ro')
end
grid on
saveas(gcf,'../../plots/lorenz_ml_traj.png')

% Build neural network
net = feedforwardnet([10 10 10]); 
net.layers{1}.transferFcn = 'logsig'; 
net.layers{2}.transferFcn = 'radbas'; 
net.layers{3}.transferFcn = 'purelin'; 

% Train
net = train(net,input.',output.');

% Choose initial condition
x0 = [
   -2.9593;
   -5.2244;
   11.6118
];

% Simulate 'true' values
[t,y] = ode45(Lorenz,t,x0);

% NN prediction
ynn(1,:)=x0;
for jj=2:length(t)
    y0=net(x0);
    ynn(jj,:)=y0.';
    x0=y0;
end

figure(2)

subplot(3,1,1)
plot(t,y(:,1),'--'); hold on
plot(t,ynn(:,1))
ylabel('y_1')
legend('y', '$\hat{y}$','Interpreter','Latex')
grid on

subplot(3,1,2)
plot(t,y(:,2),'--'); hold on
plot(t,ynn(:,2))
ylabel('y_2')
legend('y', '$\hat{y}$','Interpreter','Latex')
grid on

subplot(3,1,3)
plot(t,y(:,2),'--'); hold on
plot(t,ynn(:,2))
ylabel('y_3')
legend('y', '$\hat{y}$','Interpreter','Latex')
grid on

xlabel('t')

saveas(gcf,'../../plots/lorenz_ml_pred.png')

clear all, close all, clc

A = [1 2; 4 3]
T = [-1 0.5; 1 1]
D = [-1 0; 0 5]

N = 10;
T = 1.0;
t = linspace(0, T, N+1);
x0 = [1; 1]
x = zeros(2, N+1)
for i = 1:N+1
    x(:, i) = T*exp(D*t(i))*inv(T)*x0;
end
x

hold on
plot(t,x(1,:), 'o-')
plot(t,x(2,:), 'o-')
legend(["x_1" "x_2"])
xlabel('t')
ylabel('x')
grid()
hold off

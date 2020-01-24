% Solve linear system of equations

% x_dot = A*x
A = [1 2; 4 3]

syms x1(t) x2(t)
eqns = [diff(x1,t) == x1+2*x2, diff(x2,t) == 4*x1+3*x2];
S = dsolve(eqns);
S.x1  % ans = (C2*exp(5*t))/2 - C1*exp(-t)
S.x2  % ans = C1*exp(-t) + C2*exp(5*t)

% Solve with initial condition
x0 = [1 1];
cond = [x1(0)==1, x2(0)==1];
S = dsolve(eqns,cond);
S.x1  % ans = exp(-t)/3 + (2*exp(5*t))/3
S.x2  % ans = (4*exp(5*t))/3 - exp(-t)/3

N = 8;
tN = 0.4;
t = linspace(0, tN, N+1);

x1_sol = exp(-t)/3 + (2*exp(5*t))/3
x2_sol = (4*exp(5*t))/3 - exp(-t)/3

hold on
set(findall(gcf,'-property','FontSize'),'FontSize',15)
plot(t,x1_sol, 'o-')
plot(t,x2_sol, 'o-')
legend(["x_1" "x_2"])
xlabel('t')
ylabel('x')
grid()
hold off
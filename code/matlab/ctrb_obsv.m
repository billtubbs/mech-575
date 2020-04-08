clear all, close all, clc

m = 1;
M = 5;
L = 2;
g = -10;
d = 1;
s = -1; % pendulum up (s=1)

A = [0 1 0 0;
    0 -d/M -m*g/M 0;
    0 0 0 1;
    0 -s*d/(M*L) -s*(m+M)*g/(M*L) 0];

B = [0; 1/M; 0; s*1/(M*L)];

C = [1 0 0 0];

% Is the system (A, B) controllable?
ctrb(A, B)

% If rank = 4 it is controllable
rank(ctrb(A, B))

% Is it observable?
obsv(A,C)

% If det = 4 it is observable
det(obsv(A,C))
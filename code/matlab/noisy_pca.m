xC = [2; 1;];
sig = [2; .5;];
theta = pi/3;
R = [cos(theta) -sin(theta); sin(theta) cos(theta)];

% Generate random data
n = 10000;
X = R*diag(sig)*randn(2,n) + diag(xC)*ones(2,n);

% Compute PCA via SVD
Xavg = mean(X,2);  % row-wise mean
B = X - Xavg*ones(1,n);
[U,S,V] = svd(B/sqrt(n),'econ');

% Compute confidence intervals
theta = (0:.01:1)*2*pi;
Xstd = U*S*[cos(theta); sin(theta)];

% Plot figure
hold on
scatter(X(1,:),X(2,:),'k.','LineWidth',2)
plot(Xavg(1)+Xstd(1,:),Xavg(2) + Xstd(2,:),'r-')
plot(Xavg(1)+2*Xstd(1,:),Xavg(2) + 2*Xstd(2,:),'r-')
plot(Xavg(1)+3*Xstd(1,:),Xavg(2) + 3*Xstd(2,:),'r-')
grid();
xlabel('x');
ylabel('y');

% Now compute using pca command
% Code in book:
% [V,score,s2] = pca(X);
% disp(norm(V*score' - B))
% Raises 'Matrix dimensions must agree'

% Now compute using pca command
X = X'
[V,score,s2] = pca(X);
Xavg = mean(X,1);  % Column-wise mean
B = X - Xavg*ones(n,1);
disp(norm(V*score' - B))

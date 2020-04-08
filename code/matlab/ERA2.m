function [Ar,Br,Cr,Dr,HSVs] = ERA(YY,m,n,nin,nout,r)
% Compute a reduced order linear dynamical system model
% from impulse response data using the Eigensystem 
% Realization Algorithm (ERA).
% YY : impulse response data YY(i, j, k)
% where:
%  i refers to i-th output
%  j refers to j-th input
%  k refers to k-th timestep
% nin, nout : number of inputs and outputs
% m, n : dimensions of Hankel matrix
% r : dimensions of reduced model

Dr = YY(:,:,1);
Y = YY(:,:,2:end);
assert(size(Y,1)==nout);
assert(size(Y,2)==nin);
assert(size(Y,3)>=m+n);

for i=0:m-1
    for j=0:n-1
        H(nout*i+1:nout*(i+1),nin*j+1:nin*(j+1)) = Y(:,:,i+j+1);
        H2(nout*i+1:nout*(i+1),nin*j+1:nin*(j+1)) = Y(:,:,i+j+2);
    end
end

[U,S,V] = svd(H,'econ');
Sigma = S(1:r,1:r);
Ur = U(:,1:r);
Vr = V(:,1:r);
Ar = Sigma^(-.5)*Ur'*H2*Vr*Sigma^(-.5);
Br = Sigma^(-.5)*Ur'*H(:,1:nin);
Cr = H(1:nout,:)*Vr*Sigma^(-.5);
HSVs = diag(S);
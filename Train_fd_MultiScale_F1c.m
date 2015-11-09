function [xcindx,bkmrk,Covar,w,LML,f,err2]=Train_fd_MultiScale_F1c(X,y,hs,alphas,sigf2s,sig2,opt)
% syntax [xcindx,bkmrk,L,w,Phi,LML,f,err2]=Train_fd_MultiScale(X,y,hs,alphas,sigf2s,sig2);
% Training within the GP model with finite dimensional (fd) feature space (each center has only one scale assigned)
% determinant is computed via Cholesky decomposition of Covar
%
% input:
% X(d,n)        coordinates  of n input points in d dimensions (the design matrix)
% y(n,1)        outputs (X,y is the training set)
% hs(nsc,1)     scales (h) of Gaussian in kernel k(xi,xj)=sigf2*exp(-|xi-xj|^2/h^2)
% alphas(nsc,1) factors for space partitioning for each scale (scale a=alpha*h)
% sigf2s(nsc,1) magnitudes of the kernel k(xi,xj)=sigf2*exp(-|xi-xj|^2/h^2) for each scale
% sig2          noise level (the variance of the Gaussian noise in the training set)
% opt           flag: opt=0 (fast, without LML computations); opt=1 (slower, with LML computations)
%
% output:
% xcindx(D,1)   index of points from the training set, which are selected as centers of the reduced basis
% bkmrk(nsc+1,1)bookmark to xcindx
% Covar(D,D)    inverse covariance matrix
% w(D,1)        weights
% LML           log marginal likelihood
% f(n,1)        values of continuous fit at training points
% err2          relative L2-norm error at trainig points
%
% written by Nail Gumerov on 09/22/2014

n=numel(y);                                         %get the length of the training set 
nsc=numel(hs);                                      %get the number of scales
bkmrk=ones(nsc+1,1);                                %set bookmark to cluster center array
xcindx=zeros(n,1);                                  %set cluster center array
indx=1:n;                                           %initial set indexing

isc=0;                                              %scale index
while numel(indx~=0) && isc<nsc                     %set data structure                                                  
    isc=isc+1;
    a=alphas(isc)*hs(isc);                          %set characterictic cluster size
    Y=X(:,indx);                                    %permute                     
    ycindx=hcluster0(Y',a);                         %get permutted index to cluster centers
    nnsc=numel(ycindx);                             %get the size of the basis in the current scale
    bkmrk(isc+1)=bkmrk(isc)+nnsc;                   %set the bookmark for the next scale
    xcindx(bkmrk(isc):bkmrk(isc+1)-1)=indx(ycindx); %get true undex to cluster centers
    indx=setdiff(indx,xcindx);                      %reduce the size of the set
end;

nsc=isc;                                            %actual number of scales
bkmrk=bkmrk(1:nsc+1);                               %squeeze the bookmark

D=bkmrk(nsc+1)-1;                                   %get the dimensionality of the feature space   
xcindx=xcindx(1:D,1);                               %squeeze xcindx;
Phi=zeros(D,n);                                     %set the Phi matrix for training points

for isc=1:nsc                                       %get the Phi matrix for training points
    ind=bkmrk(isc):bkmrk(isc+1)-1;                  %current scale index
    xci=xcindx(ind);                                %index to the basis in the current scale
    Phi(ind,:)=sqrt(sigf2s(isc))*...                %get the Phi matrix for training inputs
        GaussMxnd(X(:,xci)',X',hs(isc))';
end;

Covar=Phi*Phi';                                     %get the fd kernel
Covar=sig2*eye(D)+Covar;                            %add noise regularization
L=chol(Covar,'lower');                              %get lower triangular Cholesky factorization Covar=L*L'
w=L'\(L\(Phi*y));                                   %get the weights
f=Phi'*w;                                           %get the values of continuous fit at tarining points
err2=sqrt(dot(y-f,y-f)/dot(f,f));                   %get the relative L2-norm error at trainig points

if opt==0                                           %faster computations
    LML=0;                                          %set LML to zero
elseif opt==1                                       %full computations
    LML=-0.5*(dot(y-f,y)/sig2+...                   %get the log marginal likelihood
        n*log(2*pi)+(n-D)*log(sig2))-sum(log(diag(L)));  
end;




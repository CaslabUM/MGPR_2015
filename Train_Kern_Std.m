function [L,w,LML,f,err2]=Train_Kern_Std(X,y,h,sigf2,sig2)
% syntax [L,w,LML,f,err2]=Train_Kern_Std(X,y,h,sigf2,sig2);
% Training within the standard GP model (kernel regression) using Cholesky factorization
%
% input:
% X(d,n)    coordinates  of n input points in d dimensions (the design matrix)
% y(n,1)    outputs (X,y is the training set)
% h         scale of Gaussian in kernel k(xi,xj)=sigf2*exp(-|xi-xj|^2/h^2)
% sigf2     magnitude of the kernel k(xi,xj)=sigf2*exp(-|xi-xj|^2/h^2)
% sig2      noise level (the variance of the Gaussian noise in the training set)
%
% output:
% L(n,n)    lower triangular matrix in Cholesky factorization of the regularized kernel matrix
% w(n,1)    weights
% LML       log marginal likelihood
% f(n,1)    values of continuous fit at training points
% err2      relative L2-norm error at trainig points
%
% written by Nail Gumerov on 09/08/2014

n=numel(y);                                         %get the length of the training set 
K=sigf2*GaussMxnd(X',X',h);                         %get the kernel matrix for training inputs
Ksig2=sig2*eye(n)+K;                                %add noise regularization
L=chol(Ksig2,'lower');                              %get lower triangular Cholesky factorization Ksig2=L*L'
w=L'\(L\y);                                         %get the weights
LML=-0.5*(dot(y,w)+n*log(2*pi))-sum(log(diag(L)));  %get the log marginal likelihood
f=K*w;                                              %get the values of continuous fit at tarining points
err2=sqrt(dot(y-f,y-f)/dot(f,f));                   %get the relative L2-norm error at trainig points
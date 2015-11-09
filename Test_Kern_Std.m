function [ftst,vtst,err2tst]=Test_Kern_Std(Xtst,X,L,w,h,sigf2,sig2,ytst)
% syntax [ftst,vtst,err2tst]=Test_Kern_Std(Xtst,X,L,w,h,sigf2,sig2,ytst);
% Testing within the standard GP model (kernel regression) using Cholesky factorization
%
% input:
% Xtst(d,ntst)    coordinates of ntst input points in d dimensions (the testing set)
% X(d,n)          coordinates of n input points in d dimensions (the design matrix)
% L(n,n)          lower triangular matrix in Cholesky factorization of the regularized kernel matrix (from Train_Kern_Std)
% w(n,1)          weights (from Train_Kern_Std)
% h               scale of Gaussian in kernel k(xi,x*)=sigf2*exp(-|xi-x*|^2/h^2)
% sigf2           magnitude of the kernel k(xi,x*)=sigf2*exp(-|xi-x*|^2/h^2)
% sig2            noise level (the variance of the Gaussian noise in the test set; set this to zero if there is no noise in the test set)
% ytst(ntst,1)    outputs at ntst input points (if available; otherwise put zeros(ntst,1))
%
% output:
% ftst(ntst,1)    predictive mean
% vtst(ntst,1)    predictive variance
% err2tst         relative L2-norm error at test points (makes sense only if ytst is available; for zero ytst returns 1)
%
% written by Nail Gumerov on 09/08/2014

ntst=numel(ytst);                                       %get the length of the test set 
Ktst=sigf2*GaussMxnd(X',Xtst',h);                       %get the kernel matrix for test/training inputs
ftst=Ktst*w;                                            %get the predictive mean
v1=L\Ktst'; vtst=(sigf2+sig2)*ones(ntst,1)-dot(v1,v1)'; %get the predictive variance
err2tst=sqrt(dot(ytst-ftst,ytst-ftst)/dot(ftst,ftst));  %get the relative L2-norm error at test points
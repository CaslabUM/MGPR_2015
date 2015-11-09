function [ftst,vtst,err2tst]=Test_fd_MultiScale_F1i(Xtst,X,xcindx,bkmrk,iCovar,w,hs,sigf2s,sig2,ytst)
% syntax [ftst,vtst,err2tst]=Test_fd_MultiScale(Xtst,X,xcindx,bkmrk,L,w,Phi,hs,sigf2s,sig2,ytst);
% Testing within the GP model with finite dimensional (fd) feature space (each center has only one scale assigned)
%
% input:
% Xtst(d,ntst)    coordinates of ntst input points in d dimensions (the testing set)
% X(d,n)          coordinates of n input points in d dimensions (the design matrix)
% xcindx(D,1)     indices of cluster center points
% bkmrk(nsc+1,1)  bookmark to xcindx
% iCovar(D,D)     covariance matrix (inverse to Covar from Train_fd_MultiScale_F)
% w(D,1)          weights (from Train_fd_MultiScale)
% hs(nsc,1)       scales (h) of Gaussian in kernel k(xi,x*)=sigf2*exp(-|xi-x*|^2/h^2)
% sigf2s(nsc,1)   magnitudes of the kernel k(xi,x*)=sigf2*exp(-|xi-x*|^2/h^2) for each scale
% sig2            noise level (the variance of the Gaussian noise in the test set; set this to zero if there is no noise in the test set)
% ytst(ntst,1)    outputs at ntst input points (if available; otherwise put zeros(ntst,1))
%
% output:
% ftst(ntst,1)    predictive mean
% vtst(ntst,1)    predictive variance (computed only if Covar is not empty, otherwise by default set to sig2)
% err2tst         relative L2-norm error at test points (makes sense only if ytst is available; for zero ytst returns 1)
%
% written by Nail Gumerov on 09/22/2014

ntst=numel(ytst);                                       %get the length of the test set 
nsc=numel(bkmrk)-1;                                     %get the number of scales
D=numel(xcindx);                                        %get the dimensionality of the feature space
Phitst=zeros(D,ntst);                                   %set the design matrix for test/training inputs

for isc=1:nsc                                           %get the design matrix for test/training inputs
    ind=bkmrk(isc):bkmrk(isc+1)-1;                      %current scale index
    xci=xcindx(ind);                                    %index to the basis in the current scale
    Phitst(ind,:)=sqrt(sigf2s(isc))*...                 %get the design matrix for test/training inputs
        GaussMxnd(X(:,xci)',Xtst',hs(isc))';
end;

ftst=Phitst'*w;                                         %get the predictive mean
err2tst=sqrt(dot(ytst-ftst,ytst-ftst)/dot(ftst,ftst));  %get the relative L2-norm error at test points

if numel(iCovar)==0                                     %get the predictive variance
    vtst=sig2*ones(ntst,1);                             %fast option (no variance computation)
else
    vtst=sig2*(ones(ntst,1)+dot(Phitst,iCovar*Phitst)'); %get the predictive variance
end;

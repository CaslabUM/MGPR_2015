global X y sigf2 S 
global method
X=[]; y=[]; sigf2=1; S=6; method=0;

rng(120)
sig20=0.01^2; %noise in traning data

generatenewset=1;
if generatenewset ~=0
    N=10000;
    X0=linspace(0,1,N);
    y0=sqrt(sig20)*randn(N,1);
    hX=0.1;
    N1=101; %should be odd integer
    Z1=-hX*log(linspace(exp(-0.5/hX),1,(N1+1)/2));
    X1=0.5+[-Z1((N1+1)/2-1:-1:1) Z1];
    su1=find(X0<0.5); su2=find(X0>=0.5);
    y0(su1)=y0(su1)-1;
    y0(su2)=y0(su2)+1;
    y1=sqrt(sig20)*randn(N1,1);
    su1=find(X1<0.5); su2=find(X1>=0.5);
    y1(su1)=y1(su1)-1;
    y1(su2)=y1(su2)+1;
end;

%specify training (n) and test (ntst) sets
generatenewtraining=1;
if generatenewtraining ~=0
    if Findx<5
        n=256;
        ntst=N-n;
        itrain=randi(N,1,n);%getrandint(n,N);
        itst=setdiff((1:N)',itrain);
        Xtst=X0(1,itst); ytst=y0(itst);
    else
        n=N1;
        ntst=N;
        Xtst=X0; ytst=y0;
        X=X1; y=y1;
    end;
end;
if Findx<5
    X=X0(1,itrain);  y=y0(itrain);
else
    X=X1; y=y1;
end;

%check how it works for standard GP

fprintf('\n');
disp('Standard kernel GP:')
method=0;
fprintf('\n');
disp('Training...')
tic;
sig2=0.01; h=0.1;
[opthyper,fval,exitflag,out_std]=fminsearch(@optLML_Multiscale2,[sig2 h]);
sig2=opthyper(1); h=opthyper(2);
fprintf('optimal hyperparameters: sig2 = %e h = %e, \n',sig2,h);

[L,w,LML_std,f_std,err2_std]=Train_Kern_Std(X,y,h,sigf2,sig2);
toc
disp('Testing...')
tic;
[ftst_std,vtst_std,err2tst_std]=Test_Kern_Std(Xtst,X,L,w,h,sigf2,sig2,ytst);
toc
fprintf('LML = %e  err2 = %e  err2tst = %e \n',LML_std,err2_std,err2tst_std);

fprintf('\n');
disp('Multiscale sparse GP:')
method=1;
fprintf('\n');
disp('Training...')
tic;
sig2=0.01; beta=0.5; h1=0.1; gamma=0.3;
[opthyper,fval,exitflag,out_multi]=fminsearch(@optLML_Multiscale2,[sig2 beta h1 gamma]);
sig2=opthyper(1); beta=opthyper(2); h1=opthyper(3); gamma=opthyper(4);
fprintf('optimal hyperparameters: sig2 = %e beta = %e h1 = %e gamma = %e\n',sig2,beta,h1,gamma);

alphas=gamma*ones(S,1);
s=1:S;
hss=h1*beta.^(s-1);
sigf2s=sigf2*ones(S,1);
[xcindx,bkmrk,Covar,w,LML_multi,f_multi,err2_multi]=Train_fd_MultiScale_F1c(X,y,hss,alphas,sigf2s,sig2,1);
iCovar=inv(Covar);
fprintf('RBF basis reduction from %i to %i \n',n,numel(xcindx));
toc
disp('Testing...')
tic;
[ftst_multi,vtst_multi,err2tst_multi]=Test_fd_MultiScale_F1i(Xtst,X,xcindx,bkmrk,iCovar,w,hss,sigf2s,sig2,ytst);
toc
fprintf('LML = %e  err2 = %e  err2tst = %e \n',LML_multi,err2_multi,err2tst_multi);

% generate figure
hfig=figure('position',[50 50 1200 600]); set(hfig,'Color','w');

subplot(1,2,1), eX=Xtst; eY=ftst_std; eE=sqrt(vtst_std); 
fill([eX'; flipud(eX')],[ eY+eE;  flipud((eY-eE))],'r','edgecolor','none');
alpha(0.3);
hold on, plot(X,y,'k+'),...
    plot(Xtst,ftst_std,'k'), plot(X,f_std,'k.'), xlim([0 1]),...
    title('Standard GP'), xlabel('Input, q'), ylabel('Output, y'), axis square;
subplot(1,2,2), eX=Xtst; eY=ftst_multi; eE=sqrt(vtst_multi); 
% errbar; 
fill([eX'; flipud(eX')],[ eY+eE;  flipud((eY-eE))],'r','edgecolor','none');
alpha(0.3);
hold on, plot(X,y,'k+'),...
    plot(Xtst,ftst_multi,'k'), plot(X,f_multi,'k.'), xlim([0 1]),...
    title('Multiscale GP'), xlabel('Input, q'), ylabel('Output, y'), axis square;


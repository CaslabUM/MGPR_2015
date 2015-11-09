function fLML=optLML_Multiscale2(hyper)
%optimization of the LML for multiscale
global X y sigf2 S 
global method

penalty=1000;

if method==0
    sig2=hyper(1); h=hyper(2);
    if sig2<0 || h<0
        LML=-penalty;
    else
        [~,~,LML,~,~]=Train_Kern_Std(X,y,h,sigf2,sig2);
    end;
    
elseif method==1
    sig2=hyper(1); beta=hyper(2); h1=hyper(3); gamma=hyper(4);
    if sig2<0 || beta<=0 || beta>1 || h1<0 || gamma<=0 
        LML=-(penalty +(sig2-0.001)^2 +(beta-0.5)^2+h1^2+(gamma-0.5)^2);
    else
        alphas=gamma*ones(S,1);
        s=1:S;
        hss=h1*beta.^(s-1);
        sigf2s=sigf2*ones(S,1);
        [~,~,~,~,LML,~,~]=Train_fd_MultiScale_F1c(X,y,hss,alphas,sigf2s,sig2,1);
    end;
end;
fLML=-LML;
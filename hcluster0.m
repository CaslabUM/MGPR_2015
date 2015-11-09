function xcindx=hcluster0(X,h)
%h-cluster algorithm for d-dimensional X set of size (N,d)
%written by Nail Gumerov on August 12-13, 2014
[N,~]=size(X);
xcindx=zeros(N,1);
iY=1:N;
h2=h*h; 
k=0;
while numel(iY)~=0
    k=k+1;
    xcindx(k)=min(iY);
    dist2=Dist2(X(iY,:),X(xcindx(k),:));
    su= dist2<=h2; 
    iY=setdiff(iY,iY(su));
end;
xcindx=xcindx(1:k); 

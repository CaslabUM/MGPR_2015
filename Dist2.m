function dist2=Dist2(X,x)
%square of distance between set X and point x
[N,d]=size(X);
dist2=zeros(N,1);
for j=1:d
    dist2=dist2+(X(:,j)-x(j)).^2;
end;
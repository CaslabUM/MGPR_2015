function Mx=GaussMxnd(src,rec,h)
%gaussian kernel matrix in n dimensions
si=size(src); n=si(1); d=si(2);
Mx=ones(numel(rec)/d,n);
for j=1:d
    Mx1=GaussMx(src(:,j),rec(:,j),h); Mx=Mx.*Mx1;
end;

% si=size(src); n=max(si); d=min(si);
% Mx=ones(numel(rec)/d,n);
% if si(1)==n
%     for j=1:d
%         Mx1=GaussMx(src(:,j),rec(:,j),h); Mx=Mx.*Mx1;
%     end;
% else
%     for j=1:d
%         Mx1=GaussMx(src(j,:),rec(j,:),h); Mx=Mx.*Mx1;
%     end;
% end;
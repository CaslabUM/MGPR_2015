function Mx=GaussMx(src,rec,h)
[S1,S2]=ndgrid(rec,src);
Mx=exp(-(S1-S2).^2/h^2);

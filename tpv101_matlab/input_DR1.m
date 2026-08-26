% SCEC validation problem with DR friction

refine = 1; % refinement multiplier

nx = 512*refine; % grid points in x
ny = 384*refine; % grid points in y

x0 = 0; % hypocenter, km
y0 = -7.5; % hypocenter, km
h = 0.1/refine; % grid spacing, km

% coordinates centered about (x0,y0) with point at (x0,y0)
% (one point more in x>x0 and y>y0)

x = [0:nx-1]*h; % x=[0:(nx-1)*h]
x = x0+x-0.5*max(x); % center x about x0
x = x+0.5*h; % shift by h/2 to place point at x=x0
y = [0:ny-1]*h;
y = y0+y-0.5*max(y);
y = y+0.5*h;

Wx = 15; % half-length of fault
wx = 3; % width of transition region in x-direction
Wy = 7.5; % half-width of fault
wy = 3; % width of transition region in y-direction

% smooth version of boxcar, used to set a(x,y)

Bx = boxcar(x-x0,Wx,wx);
By = boxcar(y-y0,Wy,wy);
B = zeros(nx,ny);
for ix=1:nx
  for iy=1:ny
   B(ix,iy) = Bx(ix)*By(iy);
  end
end

% friction

f0 = zeros(nx,ny)+0.6;
V0 = zeros(nx,ny)+1e-6; % m/s
a = zeros(nx,ny)+0.008+0.008*(1-B);
b = zeros(nx,ny)+0.012;
L = zeros(nx,ny)+0.02; % m

% initial stress (sx,sy,sz) and state (Q)

sx = zeros(nx,ny)+75; % MPa
sy = zeros(nx,ny); % MPa
sz = zeros(nx,ny)-120; % MPa
Vini = 1e-12; % m/s
Q = (L./V0).*exp((-sx./sz-f0-a.*log(Vini./V0))./b); % s

return

% visualize fields and demonstrate load

pcolor(x,y,a'),shading flat,colorbar,axis image
pcolor(x,y,log10(Q)'),shading flat,colorbar,axis image

T = 1; R = 3; S = zeros(nx,ny);
for t=[0:0.1*T:T]
  for ix=1:nx
    for iy=1:ny
      S(ix,iy) = 80*load_DR1(t,x(ix)-x0,y(iy)-y0,T,R);
    end
  end
  pcolor(x,y,(sx+S)'),shading flat,colorbar,axis image
  pause
end

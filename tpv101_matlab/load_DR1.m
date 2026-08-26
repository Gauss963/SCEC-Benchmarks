function S = load_DR1(t,x,y,T,R)

  % assuming scalar x,y,t; x0=0,y=0; and unit maximum amplitude
  
  r = sqrt(x^2+y^2);
  
  if r>=R
    F = 0;
  else
    F = exp(r^2/(r^2-R^2));
  end
  
  if t<=0
    G = 0;
  elseif t<T
    G = exp((t-T)^2/(t*(t-2*T)));
  else
    G = 1;
  end
    
  S = F*G;
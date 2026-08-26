function f = boxcar(x,W,w)
  
  n = length(x);
  f = zeros(1,n);
  
  for i=1:n

    X = abs(x(i));

    if X<=W
      F = 1;
    elseif X<W+w
      F = 0.5*(1+tanh(w/(X-W-w)+w/(X-W)));
    else
      F = 0;
    end
    
    f(i) = F;
    
  end
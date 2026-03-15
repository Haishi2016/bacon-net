# Feynman Triage Shortlist

## Priority 1: Hard Failures
- I.6.2a [exponential] error=optimizer got an empty parameter list formula=exp(-theta**2/2)/sqrt(2*pi)
- I.39.11 [rational] error=unsupported operand type(s) for -: 'FunctionClass' and 'One' formula=1/(gamma-1)*pr*V
- I.43.43 [rational] error=unsupported operand type(s) for -: 'FunctionClass' and 'One' formula=1/(gamma-1)*kb*v/A
- I.47.23 [sqrt-radical] error=unsupported operand type(s) for *: 'FunctionClass' and 'Symbol' formula=sqrt(gamma*pr/rho)

## Priority 2: Rational Near-Misses
- I.10.7 test=0.9833 val=0.9848 formula=m_0/sqrt(1-v**2/c**2)
- II.13.23 test=0.9833 val=0.9848 formula=rho_c_0/sqrt(1-v**2/c**2)
- I.16.6 test=0.8904 val=0.8550 formula=(u+v)/(1+u*v/c**2)
- II.38.14 test=0.8900 val=0.9158 formula=Y/(2*(1+sigma))
- I.14.4 test=0.8638 val=0.8464 formula=1/2*k_spring*x**2
- II.8.31 test=0.8638 val=0.8464 formula=epsilon*Ef**2/2
- I.34.14 test=0.8589 val=0.8626 formula=(1+v/c)/sqrt(1-v**2/c**2)*omega_0

## Priority 3: Trig Candidates
- I.29.16 test=0.2387 val=0.2014 formula=sqrt(x1**2+x2**2-2*x1*x2*cos(theta1-theta2))
- I.50.26 test=0.1181 val=0.1258 formula=x1*(cos(omega*t)+alpha*cos(omega*t)**2)
- III.15.12 test=0.1156 val=0.1637 formula=2*U*(1-cos(k*d))
- I.37.4 test=0.0990 val=0.1256 formula=I1+I2+2*sqrt(I1*I2)*cos(delta)
- II.15.4 test=0.0641 val=0.0287 formula=-mom*B*cos(theta)
- II.15.5 test=0.0641 val=0.0287 formula=-p_d*Ef*cos(theta)
- I.30.3 test=0.0494 val=0.1080 formula=Int_0*sin(n*theta/2)**2/sin(theta/2)**2
- I.18.12 test=0.0057 val=-0.0062 formula=r*F*sin(theta)
- I.12.11 test=0.0008 val=0.0426 formula=q*(Ef+B*v*sin(theta))
- III.17.37 test=-0.0227 val=-0.0297 formula=beta*(1+alpha*cos(theta))

## Priority 4: Exponential Candidates
- III.14.14 test=0.2095 val=0.1912 formula=I_0*(exp(q*Volt/(kb*T))-1)
- I.6.2 test=0.1738 val=-0.0750 formula=exp(-(theta/sigma)**2/2)/(sqrt(2*pi)*sigma)
- I.41.16 test=-0.4383 val=-0.2714 formula=h/(2*pi)*omega**3/(pi**2*c**2*(exp((h/(2*pi))*omega/(kb*T))-1))
- I.40.1 test=-0.6418 val=-0.5563 formula=n_0*exp(-m*g*x/(kb*T))
- III.4.32 test=-1.2044 val=-1.1085 formula=1/(exp((h/(2*pi))*omega/(kb*T))-1)
- III.4.33 test=-2.7906 val=-2.8422 formula=(h/(2*pi))*omega/(exp((h/(2*pi))*omega/(kb*T))-1)
- II.35.18 test=-3.0928 val=-3.4646 formula=n_0/(exp(mom*B/(kb*T))+exp(-mom*B/(kb*T)))
- I.6.2b test=-9.4341 val=-10.4218 formula=exp(-((theta-theta1)/sigma)**2/2)/(sqrt(2*pi)*sigma)

## Group Snapshot
- rational: total=58 completed=56 pass=5 mean_test_r2=-0.1651
- trig: total=16 completed=16 pass=0 mean_test_r2=-0.0749
- exponential: total=9 completed=8 pass=0 mean_test_r2=-2.1523
- pure-product: total=8 completed=8 pass=2 mean_test_r2=-6.1605
- sqrt-radical: total=5 completed=4 pass=0 mean_test_r2=-0.9163
- inverse-trig: total=2 completed=2 pass=0 mean_test_r2=-1.3541
- additive-polynomial: total=1 completed=1 pass=0 mean_test_r2=-2.3379
- sum-of-products: total=1 completed=1 pass=0 mean_test_r2=-477.1011

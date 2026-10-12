import numpy as np, math, warnings
warnings.filterwarnings("ignore")
from opt_core import *
E=2.1e8
print("=== A. CỘT ===")
for kind,kw in [("circle",{}),("rect",{"ratio":0.5}),("ibeam",{"h":0.3}),("tube",{"ratio":0.05}),("custom",dict(n=3.0,alpha=1e-3,gamma=1.0,beta=1e-2))]:
    fam=make_family(kind,**kw)
    for bc in ["pinned","cantilever","clamped"]:
        L=5.0; V=L*2e-3
        r=optimize_column(L,fam,E,bc,V=V,run_numeric=True,N_num=100,iters_num=400)
        # perturbation test: random volume-preserving perturbations of the FEM profile must not raise P
        N=400; xe=np.linspace(0,L,N+1); xm=.5*(xe[1:]+xe[:-1]); Le=np.diff(xe)
        A0=np.interp(xm,r.x,r.A_opt)
        P0,_=column_buckling_fem(Le,E*fam.I(A0),bc)
        rng=np.random.default_rng(0); worst=0
        for _ in range(40):
            d=1+0.03*rng.standard_normal(N); d=np.convolve(d,np.ones(9)/9,'same')
            A1=A0*d; A1*=V/np.sum(A1*Le)
            P1,_=column_buckling_fem(Le,E*fam.I(A1),bc); worst=max(worst,P1/P0-1)
        print(f"{kind:7s} n={fam.n:g} {bc:10s} ratio_closed={r.ratio_closed:.4f} fem={r.ratio_fem:.4f} OC={r.P_numeric/r.P_uniform:.4f}  worst-perturb dP/P={worst:+.2e}")

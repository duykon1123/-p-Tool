import numpy as np, math, warnings, time
warnings.filterwarnings("ignore")
from scipy.optimize import minimize
from opt_core import *
E=2.1e8
print("=== B. DẦM ===")
def indep_beam(prob, starts=2):
    model,xs,ctrl=build_beam_model(prob); ne=model.n_el; st=OptSettings(fam=prob.fam,E=prob.E,sigma_a=prob.sigma_a,check_buckling=False,A_min=prob.A_min,A_max=prob.A_max,
        disp_limits=[(3*ctrl+1,prob.delta_max)] if prob.delta_max else [])
    Lb=np.ones(ne)
    def ev(y):
        A=np.exp(y); U=model.solve(A); N,M=model.forces(U,A); return A,U,N,M
    def cons(y):
        A,U,N,M=ev(y); u=utilization(A,N,M,st,Lb)
        c=list(1-u)
        for d,l in st.disp_limits: c.append(1-abs(U[d])/l)
        return np.array(c)
    best=None
    for s in range(starts):
        rng=np.random.default_rng(s)
        y0=np.log(np.full(ne,prob.A_max*0.05*(1+s)))+0.05*rng.standard_normal(ne)
        r=minimize(lambda y:np.sum(np.exp(y)*model.L),y0,jac=lambda y:np.exp(y)*model.L,method="SLSQP",
            bounds=[(math.log(prob.A_min),math.log(prob.A_max))]*ne,constraints=[{"type":"ineq","fun":cons}],options=dict(maxiter=400,ftol=1e-12))
        if np.min(cons(r.x))>-1e-5 and (best is None or r.fun<best): best=r.fun
    return best
fam=make_family("ibeam",h=0.4); L=8.0
cases={
 "ss P+q":BeamProblem(L=L,bc="ss",udls=[(20.0,0,L)],point_loads=[(100.0,L/2)],fam=fam,delta_max=0.02,x_ctrl=L/2,sigma_a=2.1e5,A_min=2e-4,n_elem=30),
 "ss stress-gov":BeamProblem(L=L,bc="ss",udls=[(20.0,0,L)],point_loads=[],fam=fam,delta_max=0.05,x_ctrl=L/2,sigma_a=1.6e5,A_min=2e-4,n_elem=30),
 "cantilever":BeamProblem(L=4.0,bc="cantilever",udls=[(10.0,0,4.0)],point_loads=[(30.0,4.0)],fam=make_family("rect",ratio=0.5),delta_max=0.02,x_ctrl=4.0,sigma_a=2.1e5,A_min=2e-4,n_elem=30),
 "fixed_pinned":BeamProblem(L=L,bc="fixed_pinned",udls=[(25.0,0,L)],point_loads=[],fam=fam,delta_max=0.01,x_ctrl=L/2,sigma_a=2.1e5,A_min=2e-4,n_elem=30),
 "fixed_fixed":BeamProblem(L=10.0,bc="fixed_fixed",udls=[(30.0,0,10.0)],point_loads=[],fam=fam,delta_max=0.015,x_ctrl=5.0,sigma_a=2.1e5,A_min=5e-4,n_elem=30),
}
for k,p in cases.items():
    t=time.time(); r=optimize_beam(p); 
    ref=indep_beam(p)
    vc=r.vol_closed
    print(f"{k:14s} tool vol={r.vol_opt:.5f} feasible={r.opt.feasible} closed={vc if vc is None else round(vc,5)} indepSLSQP={ref if ref is None else round(ref,5)} uniform={r.vol_uniform:.5f} ({time.time()-t:.1f}s)")

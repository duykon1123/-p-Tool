import numpy as np, math, warnings, time
warnings.filterwarnings("ignore")
from scipy.optimize import minimize
from opt_core import *
E=2.1e8
def indep(inp, starts=3):
    model,st,mem=build_structure(inp); ne=model.n_el; Lb=st.L_buck
    def cons(y):
        A=np.exp(y); U=model.solve(A); N,M=model.forces(U,A); u=utilization(A,N,M,st,Lb)
        c=list(1-u)+[1-abs(U[d])/l for d,l in st.disp_limits]; return np.array(c)
    best=None
    for s in range(starts):
        rng=np.random.default_rng(s)
        y0=np.log(np.full(ne,inp.A_max*[0.02,0.1,0.3][s]))+0.1*rng.standard_normal(ne)
        r=minimize(lambda y:st.rho*np.sum(np.exp(y)*model.L),y0,jac=lambda y:st.rho*np.exp(y)*model.L,method="SLSQP",
            bounds=[(math.log(st.A_min),math.log(st.A_max))]*ne,constraints=[{"type":"ineq","fun":cons}],options=dict(maxiter=600,ftol=1e-13))
        ok=np.min(cons(r.x))>-1e-6
        if ok and (best is None or r.fun<best): best=r.fun
    return best
def report(name,inp):
    t=time.time(); r=optimize_frame(inp); dt=time.time()-t
    # independent feasibility re-evaluation
    m=r.model; A=r.opt.A; U=m.solve(A); N,M=m.forces(U,A); u=utilization(A,N,M,build_structure(inp)[1],build_structure(inp)[1].L_buck)
    ref=indep(inp)
    print(f"{name:22s} tool={r.weight:9.4f} feas={r.opt.feasible} maxU={u.max():.6f} indep={ref if ref is None else round(ref,4)} gap={None if ref is None else round((r.weight/ref-1)*100,2)}% uniform={r.weight_uniform:.4f} saving={r.saving*100:.1f}% ({dt:.1f}s) KKTres={r.opt.EL_cv}")
fam=make_family("tube",ratio=0.05)
# Warren truss 4 panels
nodes=[(0,0),(3,0),(6,0),(9,0),(12,0),(1.5,2.5),(4.5,2.5),(7.5,2.5),(10.5,2.5)]
mem=[(0,1),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),(0,5),(5,1),(1,6),(6,2),(2,7),(7,3),(3,8),(8,4)]
for bk in (False,True):
    report(f"Warren buck={bk}",StructInput(nodes,mem,[(0,True,True,False),(4,False,True,False)],[(5,0,-50,0),(6,0,-50,0),(7,0,-50,0),(8,0,-50,0)],[],"truss",fam,E=E,sigma_a=2.1e5,check_buckling=bk,A_min=2e-4,disp_limits=[(2,"uy",0.03)]))
# 10-bar
inch=0.0254;kip=4.4482216;ksi=6894.757
nd=[(720,360),(720,0),(360,360),(360,0),(0,360),(0,0)];nd=[(x*inch,y*inch) for x,y in nd]
mm=[(2,4),(0,2),(3,5),(1,3),(2,3),(0,1),(3,4),(2,5),(1,2),(0,3)]
lim=[(n,d,2*inch) for n in range(4) for d in("ux","uy")]
report("10-bar",StructInput(nd,mm,[(4,True,True,False),(5,True,True,False)],[(1,0,-100*kip,0),(3,0,-100*kip,0)],[],"truss",fam,E=1e4*ksi,sigma_a=25*ksi,rho=0.1*kip/inch**3,check_buckling=False,A_min=0.1*inch**2,A_max=1.0,disp_limits=lim))
# portal frame
fam2=make_family("ibeam",h=0.4)
report("Portal I h=.4",StructInput([(0,0),(0,4),(3,5),(6,4),(6,0)],[(0,1),(1,2),(2,3),(3,4)],[(0,True,True,True),(4,True,True,True)],[(1,15,0,0)],[0,-20,-20,0],"frame",fam2,E=E,sigma_a=2.1e5,check_buckling=True,A_min=2e-4,disp_limits=[(2,"uy",0.02)],segments=2))
fam3=make_family("rect",ratio=0.5)
report("2-story rect",StructInput([(0,0),(6,0),(0,3.5),(6,3.5),(0,7),(6,7)],[(0,2),(1,3),(2,3),(2,4),(3,5),(4,5)],[(0,True,True,True),(1,True,True,True)],[(2,20,0,0),(4,10,0,0)],[0,0,-25,0,0,-20],"frame",fam3,E=E,sigma_a=2.1e5,check_buckling=True,A_min=2e-4,disp_limits=[(4,"ux",0.03)],segments=2))

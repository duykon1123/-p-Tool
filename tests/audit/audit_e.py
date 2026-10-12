import numpy as np, math, warnings
warnings.filterwarnings("ignore")
from scipy.optimize import minimize
from opt_core import *
inch=0.0254;kip=4.4482216;ksi=6894.757
nd=[(720,360),(720,0),(360,360),(360,0),(0,360),(0,0)];nd=[(x*inch,y*inch) for x,y in nd]
mm=[(2,4),(0,2),(3,5),(1,3),(2,3),(0,1),(3,4),(2,5),(1,2),(0,3)]
lim=[(n,d,2*inch) for n in range(4) for d in("ux","uy")]
inp=StructInput(nd,mm,[(4,True,True,False),(5,True,True,False)],[(1,0,-100*kip,0),(3,0,-100*kip,0)],[],"truss",make_family("tube",ratio=0.05),E=1e4*ksi,sigma_a=25*ksi,rho=0.1*kip/inch**3,check_buckling=False,A_min=0.1*inch**2,A_max=1.0,disp_limits=lim)
model,st,mem=build_structure(inp); ne=10
def cons(y):
    A=np.exp(y);U=model.solve(A);N,M=model.forces(U,A);u=utilization(A,N,M,st,st.L_buck)
    return np.array(list(1-u)+[1-abs(U[d])/l for d,l in st.disp_limits])
res=[]
for s in range(40):
    rng=np.random.default_rng(s)
    y0=np.log(st.A_min)+rng.uniform(0,np.log(100*inch**2/st.A_min),ne) if s%2 else np.log(np.full(ne,rng.uniform(0.002,0.05)))
    r=minimize(lambda y:st.rho*np.sum(np.exp(y)*model.L)/kip,y0,method="SLSQP",bounds=[(math.log(st.A_min),0)]*ne,constraints=[{"type":"ineq","fun":cons}],options=dict(maxiter=800,ftol=1e-14))
    if np.min(cons(r.x))>-1e-6: res.append(r.fun/ (1) *1.0)
res=np.array(res)/ (1.0)
print("n feasible",len(res)," min lb:",res.min()*kip/kip/ (1) , " sorted unique:",np.unique(np.round(res/ (1)*1,1))[:8])

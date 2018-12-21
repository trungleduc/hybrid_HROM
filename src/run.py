

from PMFF import *
from dolfin import *

savedir = "results_error"
filepod=File(savedir+"/error.pvd")
myfile_pod=open(savedir+"/error.pvd","a",0)

mesh=Mesh("mesh0.xml")

Pmax=.01
DeltaP=-0.002
omega=16.
tl=0.25
alpha=1.
T = Expression(("0.",'t <= tl ? Pmax*t/tl : alpha*Pmax-DeltaP*sin(omega*pi*t)'), t=0.0, tl=tl,Pmax=Pmax,DeltaP=DeltaP,omega=omega,alpha=alpha)
Kre_list=np.arange(100.,201.,10.)
for K_re in Kre_list:
	problem = nonlinear_dynamics(mesh,K_re,T)
	problem.variational_formulation_solveur()
podlist = [100,150,200]
lz = 0.2
npod =21
nrid = 50
data = data_creator(mesh,lz,podlist)
data.do_svd()
Vpod = pod(mesh,lz,npod,podlist)
Vpod.save_reduced_basis()
RID_construction = rid(mesh,lz,nrid,podlist)
for K_re in Kre_list:
	run_pmff = PMFF_solver(mesh,K_re,T)
	run_pmff.variational_pmff_solveur()
	myfile_pod.write("%s %s \n" % (K_re, run_pmff.data_err))
	myfile_pod.close
	

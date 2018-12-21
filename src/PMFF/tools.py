

import scipy as sci
import scipy.linalg as lina
import numpy as np
import scipy.sparse as sp
from dolfin import *


def petsc_csr(A):
	A_mat = as_backend_type(A).mat()
	A_sparray = sp.csr_matrix(A_mat.getValuesCSR()[::-1], shape = A_mat.size)
	return A_sparray

def deim(V):
    r = V[:,0]
    i = sci.argmax(abs(r))
    F=[i]
    P = sci.zeros(sci.shape(V))
    P[i,0] = 1.
    for k in sci.arange(1,sci.shape(V)[1],1):
        gamma =    lina.solve( V[F,:k] ,    V[F,k] )
        r = V[:,k] - V[:,:k].dot(gamma);
        i = sci.argmax(abs(r))
        F = sci.append(F,i)
        P[i,k] = 1
    return P,F
    
def iz_dof_struc(meshiz,lz):
	class IZ(SubDomain):
		def inside(self, x, on_boundary):
			return True  if (x[0]<=lz)  else False

	VF = VectorFunctionSpace(meshiz, 'CG', 1)
	duf, du_tf = TrialFunction(VF), TestFunction(VF)
	subdomainsiz = CellFunction('size_t', meshiz, 0)
	subdomainsiz.set_all(0)
	izone = IZ()
	izone.mark(subdomainsiz, 1) 
	dx_f =  Measure('dx', domain=meshiz, subdomain_data=subdomainsiz)
	A_test1= assemble(inner(duf, du_tf)*dx_f(1))
	Amatrx_sp = petsc_csr(A_test1)
	Amatrx1 = Amatrx_sp.diagonal()

	dofinside=[]
	dofoutside=[]
	for i in range (np.shape(Amatrx1)[0]):
		if Amatrx1[i] != 0 :
			dofinside.append(i)
	for i in range (np.shape(Amatrx1)[0]):
		if Amatrx1[i] == 0 :
			dofoutside.append(i)

	return dofinside,dofoutside,subdomainsiz
	
def iz_dof_stress(meshiz,lz):
	class IZ(SubDomain):
		def inside(self, x, on_boundary):
			return True  if (x[0]<=lz)  else False

	VF = FunctionSpace(meshiz, 'CG', 1)
	duf, du_tf = TrialFunction(VF), TestFunction(VF)
	subdomainsiz = CellFunction('size_t', meshiz, 0)
	subdomainsiz.set_all(0)
	izone = IZ()
	izone.mark(subdomainsiz, 1) 
	dx_f =  Measure('dx', domain=meshiz, subdomain_data=subdomainsiz)
	A_test1= assemble(inner(duf, du_tf)*dx_f(1))
	Amatrx1= A_test1.array()

	dofinside=[]
	dofoutside=[]
	for i in range (len(Amatrx1[1])):
		if Amatrx1[i,i] != 0 :
			dofinside.append(i)
	for i in range (len(Amatrx1[1])):
		if Amatrx1[i,i] == 0 :
			dofoutside.append(i)

	return dofinside,dofoutside,subdomainsiz
	  

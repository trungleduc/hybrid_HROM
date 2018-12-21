import scipy as sci
import scipy.linalg as lina
import scipy.sparse.linalg as splina
import time as tm
from tools import *
from dolfin import *
import numpy as np
import math
import sympy 
import matplotlib.pyplot as plt    
from numpy import linalg as LA
import os 
import shutil
import pylab as pl
import scipy.sparse as sp

# Subdomain for load application
class Top(SubDomain):
	def set_length(self,self_h,self_l,self_lf):
		self.h = self_h
		self.l = self_l
		self.lf = self_lf
	def inside(self, x, on_boundary):
		return (near(x[1], self.h) and between(x[0], (self.lf, self.l)))  
# Inclusion domain
class Omega0(SubDomain):
	def inside(self, x, on_boundary):
		return True if (x[0]-0.1)*(x[0]-0.1)+(x[1]-0.05)*(x[1]-0.05) <0.04**2+0.000001 else False

# Class for nonlinear dynamic problem
class PMFF_solver(object):
	def __init__(self,meshInput,K_input,f_input):
		# save dir
		self.set_savedir(K_input)
		# Mesh
		self.mesh = meshInput	
		# data location
		# Parameter
		self.set_parameter(K_input)
		#Create function spaces
		self.create_fullmesh_functionspace()	
		# MeshFunctions and Measures for loading and subdomain
		self.set_fullmesh_measures()
		# File to stock data
		self.create_data_file(K_input)
		#Loading 
		self.set_loading(f_input)
		self.load_reduced_basis_RID()
		# Dirichlet BC
		self.create_submesh(self.mesh)
		self.create_submesh_measures()
		self.create_submesh_functionspace()
		self.define_bc_sub()
		self.K_function()

		
	def set_savedir(self,K_re):
		self.cwd = os.getcwd()
		self.podDirectory = "POD_Data"
		self.ssDirectory = "Offline_Data"
		self.ridDirectory = "RID_Data"
		self.rpodDirectory = "RPOD_Data"
		savedir = "result_PMFF/K_%01d" %K_re
		if os.path.isdir(savedir):
		  shutil.rmtree(savedir)
		self.file_online = File(savedir+"/online/u_online.pvd")
		self.file_stress = File(savedir+"/stress/stress_xx.pvd")
	
	def create_fullmesh_functionspace(self):
		self.V = VectorFunctionSpace(self.mesh, 'CG', 1)
		self.V0 = FunctionSpace(self.mesh, 'DG', 1)
		self.VS = TensorFunctionSpace(self.mesh, 'CG', 1)
		self.Vq1 = FunctionSpace(self.mesh,  'CG', 1) 
		#Solution, test and trial functions
		self.du, self.du_t = TrialFunction(self.V), TestFunction(self.V)
		self.uk = Function(self.V)
		self.u_pred,self.v_pred=Function(self.V),Function(self.V)
		self.u_n,self.v_n,self.a_n = Function(self.V),Function(self.V),Function(self.V)
		self.u,self.v,self.a = Function(self.V),Function(self.V),Function(self.V)
	
		self.d = self.uk.geometric_dimension()
		self.I = Identity(self.d)
	
	def create_data_file(self,Kre_input):
		time1 = pl.arange(0,self.t_max1, self.dt)
		self.Q_dof_PMFF = np.zeros(shape=(self.u.vector().size(),len(time1)))
		self.Q_dof = np.load(self.cwd+"/"+self.ssDirectory+'/Q_dof_%01d.npy' %Kre_input)
	def set_parameter(self,Kre_input):
		#beam height
		self.h = 0.1
		self.l = 1.
		# loading length
		self.lf=0.2 
		# length of interest zone
		self.lz=0.2 
		#newmark parameter
		self.theta=0.5
		self.gamma = 0.5
		self.beta = 0.25
		#time step
		self.dt=0.0025
		self.t_max1 = 1.
		# parameters of steel
		self.rho=8000./1000000000.
		self.K, self.nu = 200.0, 0.3
		self.K_re = Kre_input
		self.mu, self.lmbda = Constant(self.K/(2*(1 + self.nu))), Constant(self.K*self.nu/((1 + self.nu)*(1 - 2*self.nu)))
		self.mu_re, self.lmbda_re = Constant(self.K_re/(2*(1 + self.nu))), Constant(self.K_re*self.nu/((1 + self.nu)*(1 - 2*self.nu))) 
	
	def set_loading(self,f_input):
		self.T = f_input
		
	def set_fullmesh_measures(self):
		# Initialize sub-domain instances
		self.top = Top()
		self.top.set_length(self.h,self.l,self.lf)
		# Initialize mesh function for loading
		self.boundaries = FacetFunction("size_t", self.mesh)
		self.boundaries.set_all(0)
		self.top.mark(self.boundaries, 2)
		# Load mesh fuction from file for inclusion marker
		self.Ydomains = MeshFunction('size_t', self.mesh, 'mesh_physical_region.xml')
		self.ds = Measure('ds', domain=self.mesh, subdomain_data=self.boundaries)
		self.dx = Measure('dx', domain=self.mesh, subdomain_data=self.Ydomains)
	
	def K_function(self):
		self.mu_f = Function(self.V0)
		self.lmbda_f = Function(self.V0)
		self.mu_values = [self.K/(2*(1 + self.nu)), self.K_re/(2*(1 + self.nu))]  # values of k in the two subdomains
		self.lmbda_values = [self.K*self.nu/((1 + self.nu)*(1 - 2*self.nu)), self.K_re*self.nu/((1 + self.nu)*(1 - 2*self.nu))]
		for cell_no in range(len(self.Ydomains.array())):
			subdomain_no = self.Ydomains.array()[cell_no]
			self.mu_f.vector()[cell_no] = self.mu_values[subdomain_no]
			self.lmbda_f.vector()[cell_no] = self.lmbda_values[subdomain_no]

	def load_reduced_basis_RID(self):
		V_POD_load = np.load(self.cwd+"/"+self.podDirectory+'/V_POD.npy')
		self.V_POD = sp.csr_matrix(V_POD_load)
		VR_POD_load = np.load(self.cwd+"/"+self.rpodDirectory+'/VR_POD.npy')
		self.VR_POD = sp.csr_matrix(VR_POD_load)
		self.ZtZ= np.load(self.cwd+"/"+self.rpodDirectory+'/ZtZ.npy')
		self.N = np.shape(self.V_POD)[1]
		self.Ndof = np.shape(self.V_POD)[0]
		self.Nrdof=np.shape(self.VR_POD)[0]
		self.ZtZ_sp=sp.spdiags(np.diag(self.ZtZ),0,np.shape(self.ZtZ)[0],np.shape(self.ZtZ)[1])
	
	def create_submesh(self,mesh):
		self.subdomains = MeshFunction('size_t',mesh, self.cwd+"/"+self.ridDirectory+"/subdomain.xml")
		self.submesh = SubMesh(mesh, self.subdomains, 1)
	
	def create_submesh_measures(self):
		self.mydomains = CellFunction('size_t', self.submesh)
		self.mydomains.set_all(0)
		self.subdomain_y = Omega0()
		self.subdomain_y.mark(self.mydomains, 1)
		#plot(self.mydomains,interactive=True)
		self.dx_subdomain =Measure('dx', domain=self.submesh, subdomain_data=self.mydomains)
		self.sub_boundaries = FacetFunction("size_t", self.submesh)
		self.sub_boundaries.set_all(0)
		top = Top()
		top.set_length(self.h,self.l,self.lf)
		top.mark(self.sub_boundaries, 2)
		self.ds_subdomain = Measure('ds', domain=self.submesh, subdomain_data=self.sub_boundaries)
		
	def create_submesh_functionspace(self):
		self.Vt = VectorFunctionSpace(self.submesh, "Lagrange", 1)
		self.du_sub, self.du_t_sub = TrialFunction(self.Vt), TestFunction(self.Vt)
	
	def define_bc_sub(self):
		left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
		c0 = Expression(("0.0", "0.0"))
		self.bc_sub = DirichletBC(self.Vt, c0, left)
	
	def FormR(self,u_f,u_test,u_prediction,force):
		du_f=nabla_grad(u_f)
		du_te=nabla_grad(u_test)
		P_int=(self.I+du_f)*(self.lmbda*tr(0.5*(du_f+du_f.T)+du_f.T*du_f)*self.I+2*self.mu*(0.5*(du_f+du_f.T)+du_f.T*du_f))
		P_re=(self.I+du_f)*(self.lmbda_re*tr(0.5*(du_f+du_f.T)+du_f.T*du_f)*self.I+2*self.mu_re*(0.5*(du_f+du_f.T)+du_f.T*du_f))
		Fr=(self.rho/(self.beta*self.dt*self.dt))*inner(u_f-u_prediction,u_test)*self.dx_subdomain+inner(P_int,du_te)*self.dx_subdomain(0)+inner(P_re,du_te)*self.dx_subdomain(1)-inner(force,u_test)*self.ds_subdomain(2)
		return Fr;		

			
	def JacobianR(self,u_f,u_trial,u_test):
		du_f=nabla_grad(u_f)
		du_tr=nabla_grad(u_trial)
		du_te=nabla_grad(u_test)
		dF0_int=(self.I+du_f)*(self.lmbda*tr(0.5*(du_tr+du_tr.T)+du_tr.T*du_f+du_f.T*du_tr)*self.I+2*self.mu*(0.5*(du_tr+du_tr.T)+du_tr.T*du_f+du_f.T*du_tr))
		dF1_int=du_tr*(self.lmbda*tr(0.5*(du_f+du_f.T)+du_f.T*du_f)*self.I+2*self.mu*(0.5*(du_f+du_f.T)+du_f.T*du_f))
		dF0_re=(self.I+du_f)*(self.lmbda_re*tr(0.5*(du_tr+du_tr.T)+du_tr.T*du_f+du_f.T*du_tr)*self.I+2*self.mu_re*(0.5*(du_tr+du_tr.T)+du_tr.T*du_f+du_f.T*du_tr))
		dF1_re=du_tr*(self.lmbda_re*tr(0.5*(du_f+du_f.T)+du_f.T*du_f)*self.I+2*self.mu_re*(0.5*(du_f+du_f.T)+du_f.T*du_f))
		Jr=(self.rho/(self.beta*self.dt*self.dt))*inner(u_trial,u_test)*self.dx_subdomain+(inner(dF0_int+dF1_int,du_te)*self.dx_subdomain(0)+inner(dF0_re+dF1_re,du_te)*self.dx_subdomain(1))
		return Jr;
			
	def Pstress(self,u_f):
		du_f=nabla_grad(u_f)
		P_stress=(self.I+du_f)*(self.lmbda_f*tr(0.5*(du_f+du_f.T)+du_f.T*du_f)*self.I+2*self.mu_f*(0.5*(du_f+du_f.T)+du_f.T*du_f))
		return P_stress;
					
	def variational_pmff_solveur(self):
		j=0
		M_0=assemble(self.rho*inner(self.du, self.du_t)*self.dx)
		F_0=assemble(-inner(self.Pstress(self.u),nabla_grad(self.du_t))*self.dx)
		def solve_a():
			solve(M_0, self.a_n.vector(), F_0)  
			
		time2 = pl.arange(0,self.t_max1, self.dt)
		start_online = tm.clock()
		for i, t in enumerate(time2):
			self.T.t=t
			tol = 1.0E-4
			iter = 0
			maxiter = 250
			eps = 1.0
			if i==0:
				solve_a()
			else:
				# Predictions for u^n and v^n
				self.u_pred.vector().set_local(self.u_n.vector().array() + self.dt*self.v_n.vector().array() + 0.5*self.dt**2*(1-2*self.beta)*self.a_n.vector().array())
				self.v_pred.vector().set_local(self.v_n.vector().array() + self.dt*(1-self.gamma)*self.a_n.vector().array())
				self.uk.vector().set_local(self.u_n.vector().array()) # use u_n for predicted solution at n+1 step
				self.u_pred_sub=interpolate(self.u_pred, self.Vt)
				# Solve for displacement at n+1 step
				while eps > tol and iter < maxiter:
					iter += 1
					self.uk_sub=interpolate(self.uk, self.Vt)
					A1=assemble(self.JacobianR(self.uk_sub,self.du_sub,self.du_t_sub))
					b1=-assemble(self.FormR(self.uk_sub,self.du_t_sub,self.u_pred_sub,self.T))
					self.bc_sub.apply(A1,b1)
					A1_sp=petsc_csr(A1)
					b1_sp=b1.array()
					A1_spz=self.ZtZ_sp.dot(A1_sp)
					b1_array=self.ZtZ_sp.dot(b1_sp)
					Ar1=self.VR_POD.T.dot(A1_spz.dot(self.VR_POD))
					br1=self.VR_POD.T.dot(b1_array)
					q_du=splina.spsolve(Ar1, br1)
					temps=self.V_POD.dot(q_du)
					eps = lina.norm(q_du, ord=np.Inf)
					print 'Norm:', eps
					self.u.vector().set_local(self.uk.vector().array()+temps)
					self.uk.vector().set_local(self.u.vector().array())
		
				#update displacement, vlocity and acceleration at n+1 step
				self.a_n.vector().set_local((self.u.vector().array()-self.u_pred.vector().array()) / (self.beta*self.dt**2))
				self.v_n.vector().set_local(self.v_pred.vector().array() + self.dt*self.gamma*self.a_n.vector().array())
				self.u_n.vector().set_local(self.u.vector().array())
				self.Q_dof_PMFF[:,j] =  self.u.vector().array()
				j=j+1
				self.u.rename("u_PMFF_K%01d" %self.K_re , "stress component")
				#self.file_online << self.u
				print i
		end_online = tm.clock()	
		self.onlinetime=end_online - start_online	
		A_err=self.Q_dof_PMFF-self.Q_dof
		Norm_err=np.zeros(np.shape(A_err)[1])
		for i in  range(np.shape(A_err)[1]):
			if np.linalg.norm(self.Q_dof[:,i], ord=2) !=0:
				Norm_err[i]=(np.linalg.norm(A_err[:,i], ord=2))/(np.linalg.norm(self.Q_dof[:,i], ord=2))
		self.data_err= np.sum(Norm_err)/(float(len(Norm_err)-1))
		print self.data_err	


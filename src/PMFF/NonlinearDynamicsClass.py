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

# Class for nonlinear dynamic problem
class nonlinear_dynamics(object):
	def __init__(self,meshInput, K_input,f_input):
		# save dir
		self.set_savedir(K_input)
		
		# Mesh
		self.mesh = meshInput	
		
		# Parameter
		self.set_parameter(K_input)
		
		#Create function spaces
		self.V = VectorFunctionSpace(self.mesh, 'CG', 1)
		self.VS = TensorFunctionSpace(self.mesh, 'CG', 1)
		self.Vq1 = FunctionSpace(self.mesh,  'CG', 1) 
		self.V0 =  FunctionSpace(self.mesh, 'DG', 0)
		#Solution, test and trial functions
		self.du, self.du_t = TrialFunction(self.V), TestFunction(self.V)
		self.uk = Function(self.V)
		self.u_pred,self.v_pred=Function(self.V),Function(self.V)
		self.u_n,self.v_n,self.a_n = Function(self.V),Function(self.V),Function(self.V)
		self.u,self.v,self.a = Function(self.V),Function(self.V),Function(self.V)
		self.dqu= Function(self.V)	
		
		self.d = self.uk.geometric_dimension()
		self.I = Identity(self.d)
		# MeshFunctions and Measures for loading and subdomain
		self.set_mesh_functions()
		self.set_measures()
		# File to stock data
		self.create_data_file(K_input)
		#Loading 
		self.set_loading(f_input)
		
		# Dirichlet BC
		self.bc_u = self.define_bc_u()
		self.K_function()
        # Variational formulation
		#self.set_variational_formulation() 
		
	def set_savedir(self,K_re):
		self.savedir = "results_EF/K_%01d" %K_re
		if os.path.isdir(self.savedir):
			shutil.rmtree(self.savedir)
		self.file_offline = File(self.savedir+"/offline/u_offline.pvd")
		self.file_stress = File(self.savedir+"/stress/stress_xx.pvd")
		self.file_mesh=File(self.savedir+"/mesh0.pvd")
	
	def create_data_file(self,Kre_input):
		time1 = pl.arange(0,self.t_max1, self.dt)
		self.Q_dof = np.zeros(shape=(self.u.vector().size(),len(time1)))
		#self.Qstress_tensor = np.zeros(shape=(self.u.vector().size()*2,len(time1)))
		self.Qtress_dof = np.zeros(shape=(self.u.vector().size()/2,len(time1)))
		self.Qtress1_dof = np.zeros(shape=(self.u.vector().size()/2,len(time1)))
		self.Qtress01_dof = np.zeros(shape=(self.u.vector().size()/2,len(time1)))
		self.Qtress10_dof = np.zeros(shape=(self.u.vector().size()/2,len(time1)))
		cwd = os.getcwd()
		directory = "Offline_Data"
		if not os.path.exists(directory):
			os.makedirs(directory)
		self.Q_dof_filepath = cwd+"/"+directory+"/Q_dof_%01d" %Kre_input
		self.Qtress_dof_filepath = cwd+"/"+directory+"/Qstress_%01d" %Kre_input
		self.Qtress1_dof_filepath = cwd+"/"+directory+"/Qstress1_%01d" %Kre_input
		self.Qtress01_dof_filepath = cwd+"/"+directory+"/Qstress01_%01d" %Kre_input
		self.Qtress10_dof_filepath = cwd+"/"+directory+"/Qstress10_%01d" %Kre_input
		print self.Q_dof_filepath
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
		
	def set_mesh_functions(self):
		# Initialize sub-domain instances
		self.top = Top()
		self.top.set_length(self.h,self.l,self.lf)
		# Initialize mesh function for loading
		self.boundaries = FacetFunction("size_t", self.mesh)
		self.boundaries.set_all(0)
		self.top.mark(self.boundaries, 2)
		# Load mesh fuction from file for inclusion marker
		self.Ydomains = MeshFunction('size_t', self.mesh, 'mesh_physical_region.xml')
		
	def set_measures(self):
		self.ds = Measure('ds', domain=self.mesh, subdomain_data=self.boundaries)
		self.dx = Measure('dx', domain=self.mesh, subdomain_data=self.Ydomains)
		
	def define_bc_u(self):
		left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
		c0 = Expression(("0.0", "0.0"))
		bc_u0 = DirichletBC(self.V, c0, left)
		bcs = bc_u0
		return bcs
	def K_function(self):
		self.mu_f = Function(self.V0)
		self.lmbda_f = Function(self.V0)
		self.mu_values = [self.K/(2*(1 + self.nu)), self.K_re/(2*(1 + self.nu))]  # values of k in the two subdomains
		self.lmbda_values = [self.K*self.nu/((1 + self.nu)*(1 - 2*self.nu)), self.K_re*self.nu/((1 + self.nu)*(1 - 2*self.nu))]
		for cell_no in range(len(self.Ydomains.array())):
			subdomain_no = self.Ydomains.array()[cell_no]
			self.mu_f.vector()[cell_no] = self.mu_values[subdomain_no]
			self.lmbda_f.vector()[cell_no] = self.lmbda_values[subdomain_no]
	
	def Form(self,u_f,u_test,u_prediction,force):
		du_f=nabla_grad(u_f)
		du_te=nabla_grad(u_test)
		P_int=(self.I+du_f)*(self.lmbda*tr(0.5*(du_f+du_f.T)+du_f.T*du_f)*self.I+2*self.mu*(0.5*(du_f+du_f.T)+du_f.T*du_f))
		P_re=(self.I+du_f)*(self.lmbda_re*tr(0.5*(du_f+du_f.T)+du_f.T*du_f)*self.I+2*self.mu_re*(0.5*(du_f+du_f.T)+du_f.T*du_f))
		F=(self.rho/(self.beta*self.dt*self.dt))*inner(u_f-u_prediction,u_test)*self.dx+(inner(P_int,du_te)*self.dx(0)+inner(P_re,du_te)*self.dx(1)-inner(force,u_test)*self.ds(2))
		return F;
			
	def Jacobian(self,u_f,u_trial,u_test):
		du_f=nabla_grad(u_f)
		du_tr=nabla_grad(u_trial)
		du_te=nabla_grad(u_test)
		dF0_int=(self.I+du_f)*(self.lmbda*tr(0.5*(du_tr+du_tr.T)+du_tr.T*du_f+du_f.T*du_tr)*self.I+2*self.mu*(0.5*(du_tr+du_tr.T)+du_tr.T*du_f+du_f.T*du_tr))
		dF1_int=du_tr*(self.lmbda*tr(0.5*(du_f+du_f.T)+du_f.T*du_f)*self.I+2*self.mu*(0.5*(du_f+du_f.T)+du_f.T*du_f))
		dF0_re=(self.I+du_f)*(self.lmbda_re*tr(0.5*(du_tr+du_tr.T)+du_tr.T*du_f+du_f.T*du_tr)*self.I+2*self.mu_re*(0.5*(du_tr+du_tr.T)+du_tr.T*du_f+du_f.T*du_tr))
		dF1_re=du_tr*(self.lmbda_re*tr(0.5*(du_f+du_f.T)+du_f.T*du_f)*self.I+2*self.mu_re*(0.5*(du_f+du_f.T)+du_f.T*du_f))
		J=(self.rho/(self.beta*self.dt*self.dt))*inner(u_trial,u_test)*self.dx+(inner(dF0_int+dF1_int,du_te)*self.dx(0)+inner(dF0_re+dF1_re,du_te)*self.dx(1))
		return J;	
			
	def Pstress(self,u_f):
		du_f=nabla_grad(u_f)
		P_stress=(self.I+du_f)*(self.lmbda_f*tr(0.5*(du_f+du_f.T)+du_f.T*du_f)*self.I+2*self.mu_f*(0.5*(du_f+du_f.T)+du_f.T*du_f))
		return P_stress;
				
	def variational_formulation_solveur(self):
		j=0
		M_0=assemble(self.rho*inner(self.du, self.du_t)*self.dx)
		F_0=assemble(-inner(self.Pstress(self.u),nabla_grad(self.du_t))*self.dx)
		def solve_a():
			solve(M_0, self.a_n.vector(), F_0)  
			
		time1 = pl.arange(0,self.t_max1, self.dt)
		for i, t in enumerate(time1):
			self.T.t=t
			tol = 1.0E-5
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
				# Solve for displacement at n+1 step
				while eps > tol and iter < maxiter:
					iter += 1
					A=assemble(self.Jacobian(self.uk,self.du,self.du_t))
					b=-assemble(self.Form(self.uk,self.du_t,self.u_pred,self.T))
					self.bc_u.apply(A,b)
					A1_sp=petsc_csr(A)
					b1_sp=b.array()
					q_du=splina.spsolve(A1_sp, b1_sp)
					eps = np.linalg.norm(q_du, ord=np.Inf)
					print 'Norm:', eps
					self.u.vector().set_local(self.uk.vector().array()+q_du)
					self.uk.vector().set_local(self.u.vector().array()) 
			
				#plot(self.uk, key = "alpha",mode = "displacement", title = "Damage at loading %.4f",interactive=False)
				#update displacement, vlocity and acceleration at n+1 step
				self.a_n.vector().set_local((self.u.vector().array()-self.u_pred.vector().array()) / (self.beta*self.dt**2))
				self.v_n.vector().set_local(self.v_pred.vector().array() + self.dt*self.gamma*self.a_n.vector().array())
				self.u_n.vector().set_local(self.u.vector().array())
				#stock solution to numpy narray
				stress=self.Pstress(self.u)
				stress_tensor=project(stress,self.VS)
				stress00=project(stress[0,0],self.Vq1)
				stress11=project(stress[1,1],self.Vq1)
				stress01=project(stress[0,1],self.Vq1)
				stress10=project(stress[1,0],self.Vq1)
				self.Qtress_dof[:,j]=stress00.vector().array()
				self.Qtress1_dof[:,j]=stress11.vector().array()
				self.Qtress01_dof[:,j]=stress01.vector().array()
				self.Qtress10_dof[:,j]=stress10.vector().array()
				#self.Qstress_tensor[:,j]=stress_tensor.vector().array()
				self.Q_dof[:,j] =  self.u.vector().array()
				j=j+1
				self.u.rename("u_EF_K%01d" %self.K_re , "stress component")
				self.file_offline << self.u
			print t
		np.save(self.Q_dof_filepath, self.Q_dof)
		np.save(self.Qtress_dof_filepath, self.Qtress_dof)
		np.save(self.Qtress1_dof_filepath, self.Qtress1_dof)
		np.save(self.Qtress01_dof_filepath, self.Qtress01_dof)
		np.save(self.Qtress10_dof_filepath, self.Qtress10_dof)
		
				


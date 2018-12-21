

import scipy as sci
import numpy as np
from DataCreator import *
from dolfin import *
from tools import *

class pod(data_creator):
	def __init__(self,meshInput,lzInput,numberModeInput,ssListInput):
		data_creator.__init__(self,meshInput,lzInput,ssListInput)

		# remap truncated POD matrix to full matrix
		self.create_full_pod(numberModeInput)
		# concatenate with FE shape function
		self.create_full_reduced_basis()

		
	# override load_data() of data_creator to load POD basis instead of snapshot data 	
	def load_data(self):
		self.V_POD_iz_out=np.load(self.cwd+"/"+self.podDirectory+'/V_POD_iz_out.npy')
		self.S_POD_iz_out=np.load(self.cwd+"/"+self.podDirectory+'/S_POD_iz_out.npy')
		self.S1_POD_iz_out=np.load(self.cwd+"/"+self.podDirectory+'/S1_POD_iz_out.npy')
		self.S01_POD_iz_out=np.load(self.cwd+"/"+self.podDirectory+'/S01_POD_iz_out.npy')
		self.S10_POD_iz_out=np.load(self.cwd+"/"+self.podDirectory+'/S10_POD_iz_out.npy')
	
	def create_full_pod(self,numberModeInput):
		self.V_POD_iz_out=self.V_POD_iz_out[:,:numberModeInput]
		self.N_iz_out = np.shape(self.V_POD_iz_out)[1]
		self.S_POD_iz_out=self.S_POD_iz_out[:,:self.N_iz_out]
		self.S1_POD_iz_out=self.S1_POD_iz_out[:,:self.N_iz_out]
		self.S01_POD_iz_out=self.S01_POD_iz_out[:,:self.N_iz_out]
		self.S10_POD_iz_out=self.S10_POD_iz_out[:,:self.N_iz_out]
		
		self.Ndof_iz_out = np.shape(self.V_POD_iz_out)[0]
		print 'Number of POD modes  = ', self.N_iz_out
			
		self.V_POD_iz_out_full=np.zeros(shape=(self.Ndof,self.N_iz_out))

		self.S_POD_iz_out_full=np.zeros(shape=(self.Ndof/2,self.N_iz_out))
		self.S1_POD_iz_out_full=np.zeros(shape=(self.Ndof/2,self.N_iz_out))
		self.S01_POD_iz_out_full=np.zeros(shape=(self.Ndof/2,self.N_iz_out))
		self.S10_POD_iz_out_full=np.zeros(shape=(self.Ndof/2,self.N_iz_out))
		
		for iz in range(len(self.IZ_dofoutside)):
			self.V_POD_iz_out_full[self.IZ_dofoutside[iz],:]=self.V_POD_iz_out[iz,:]
				
		for iz in range(len(self.stress_dofoutside)):
			self.S_POD_iz_out_full[self.stress_dofoutside[iz],:]=self.S_POD_iz_out[iz,:]
			
		for iz in range(len(self.stress_dofoutside)):
			self.S1_POD_iz_out_full[self.stress_dofoutside[iz],:]=self.S1_POD_iz_out[iz,:]
		
		for iz in range(len(self.stress_dofoutside)):
			self.S01_POD_iz_out_full[self.stress_dofoutside[iz],:]=self.S01_POD_iz_out[iz,:]
		
		for iz in range(len(self.stress_dofoutside)):
			self.S10_POD_iz_out_full[self.stress_dofoutside[iz],:]=self.S10_POD_iz_out[iz,:]	
		
	def create_full_reduced_basis(self):
		V_POD_iz_in_full=np.zeros(shape=(self.Ndof,len(self.IZ_dofinside)))
		for iz in range(len(self.IZ_dofinside)):
			V_POD_iz_in_full[self.IZ_dofinside[iz],iz]=1.	 
		self.V_POD=np.concatenate((V_POD_iz_in_full, self.V_POD_iz_out_full), axis=1)
		#print 'shape of V_POD :' , np.shape(self.V_POD)

	#save reduced basis to file
	def save_reduced_basis(self):
		np.save(self.cwd+"/"+self.podDirectory+'/V_POD', self.V_POD)
		np.save(self.cwd+"/"+self.podDirectory+'/S_POD_out', self.S_POD_iz_out_full)
		np.save(self.cwd+"/"+self.podDirectory+'/S1_POD_out', self.S1_POD_iz_out_full)
		np.save(self.cwd+"/"+self.podDirectory+'/S01_POD_out', self.S01_POD_iz_out_full)
		np.save(self.cwd+"/"+self.podDirectory+'/S10_POD_out', self.S10_POD_iz_out_full)
		np.save(self.cwd+"/"+self.podDirectory+'/IZ_dofoutside', self.IZ_dofoutside)
		#np.save('V_POD_out', V_POD_iz_out)
		#np.save('V_POD_out_full', V_POD_iz_out_full)
		#np.save('V_POD_in_full', V_POD_iz_in_full))
			

			



from dolfin import *
from tools import *
import os
import numpy as np
import time as tm
import csv

class data_creator(object):
	def __init__(self,meshInput, lzInput,ssListInput):
		self.POD_LIST = ssListInput
		#number of snapshot used
		self.NUMBER_OF_POD = len(self.POD_LIST)
		self.cwd = os.getcwd()
		self.podDirectory = "POD_Data"
		self.ssDirectory = "Offline_Data"
		self.ridDirectory = "RID_Data"
		self.gdim = meshInput.geometry().dim() #  The geometrical dimension.   
		self.tdim = meshInput.topology().dim() #  The topological dimension.
		self.rpodDirectory = "RPOD_Data"
		if not os.path.exists(self.podDirectory):
			os.makedirs(self.podDirectory)
		if not os.path.exists(self.ridDirectory):
			os.makedirs(self.ridDirectory)
		if not os.path.exists(self.rpodDirectory):
			os.makedirs(self.rpodDirectory)
		#load snapshot matrix
		self.load_data()
		#create set of DoFs of reduced domain
		self.create_reduced_dof_list(meshInput,lzInput)
		#number of DoFs
		self.Ndof = len(self.IZ_dofinside)+len(self.IZ_dofoutside)
		
	def load_data(self):
		self.Q_dof_full = np.load(self.cwd+"/"+self.ssDirectory+"/Q_dof_%01d.npy" %self.POD_LIST[0])
		self.S_dof_full = np.load(self.cwd+"/"+self.ssDirectory+"/Qstress_%01d.npy" %self.POD_LIST[0])		
		self.S1_dof_full = np.load(self.cwd+"/"+self.ssDirectory+"/Qstress1_%01d.npy" %self.POD_LIST[0])		
		self.S01_dof_full = np.load(self.cwd+"/"+self.ssDirectory+"/Qstress01_%01d.npy" %self.POD_LIST[0])
		self.S10_dof_full = np.load(self.cwd+"/"+self.ssDirectory+"/Qstress10_%01d.npy" %self.POD_LIST[0])				
		for i in range(1,self.NUMBER_OF_POD):
			Q_dof_full_load =np.load(self.cwd+"/"+self.ssDirectory+'/Q_dof_%01d.npy' %self.POD_LIST[i])
			self.Q_dof_full=np.concatenate((self.Q_dof_full,Q_dof_full_load), axis=1)
			S_dof_full_load =np.load(self.cwd+"/"+self.ssDirectory+'/Qstress_%01d.npy' %self.POD_LIST[i])
			self.S_dof_full=np.concatenate((self.S_dof_full,S_dof_full_load), axis=1)
			S1_dof_full_load =np.load(self.cwd+"/"+self.ssDirectory+'/Qstress1_%01d.npy' %self.POD_LIST[i])
			self.S1_dof_full=np.concatenate((self.S1_dof_full,S1_dof_full_load), axis=1)
			S01_dof_full_load =np.load(self.cwd+"/"+self.ssDirectory+'/Qstress01_%01d.npy' %self.POD_LIST[i])
			self.S01_dof_full=np.concatenate((self.S01_dof_full,S01_dof_full_load), axis=1)
			S10_dof_full_load =np.load(self.cwd+"/"+self.ssDirectory+'/Qstress10_%01d.npy' %self.POD_LIST[i])
			self.S10_dof_full=np.concatenate((self.S10_dof_full,S10_dof_full_load), axis=1)
		
	def create_reduced_dof_list(self,mesh,lz):
		self.IZ_dofinside,self.IZ_dofoutside,self.subdomains_iz=iz_dof_struc(mesh,lz)
		self.stress_dofinside,self.stress_dofoutside,self.subdomains_stress=iz_dof_stress(mesh,lz)
	
	def do_svd(self):
		time_start = tm.time()
		print 'SVD Q_dof'
		Q_dof_iz_out = np.delete(self.Q_dof_full,self.IZ_dofinside, axis=0)
		self.V_POD_iz_out,self.sigma_POD_iz_out,self.W_POD_iz_out = np.linalg.svd(Q_dof_iz_out,0)
		np.save(self.cwd+"/"+self.podDirectory+'/V_POD_iz_out', self.V_POD_iz_out)
		np.save(self.cwd+"/"+self.podDirectory+'/sigma_POD_iz_out', self.sigma_POD_iz_out)
		
		print 'SVD S_dof'
		S_dof_iz_out = np.delete(self.S_dof_full,self.stress_dofinside, axis=0)		
		self.S_POD_iz_out,self.sigmaS_POD_iz_out,self.WS_POD_iz_out = np.linalg.svd(S_dof_iz_out,0)
		np.save(self.cwd+"/"+self.podDirectory+'/S_POD_iz_out', self.S_POD_iz_out)
		np.save(self.cwd+"/"+self.podDirectory+'/sigmaS_POD_iz_out', self.sigmaS_POD_iz_out)
		
		print 'SVD S1_dof'
		S1_dof_iz_out = np.delete(self.S1_dof_full,self.stress_dofinside, axis=0)		
		self.S1_POD_iz_out,self.sigmaS1_POD_iz_out,self.WS1_POD_iz_out = np.linalg.svd(S1_dof_iz_out,0)
		np.save(self.cwd+"/"+self.podDirectory+'/S1_POD_iz_out', self.S1_POD_iz_out)
		np.save(self.cwd+"/"+self.podDirectory+'/sigmaS1_POD_iz_out', self.sigmaS1_POD_iz_out)
		
		print 'SVD S01_dof'
		S01_dof_iz_out = np.delete(self.S01_dof_full,self.stress_dofinside, axis=0)		
		self.S01_POD_iz_out,self.sigmaS01_POD_iz_out,self.WS01_POD_iz_out = np.linalg.svd(S01_dof_iz_out,0)
		np.save(self.cwd+"/"+self.podDirectory+'/S01_POD_iz_out', self.S01_POD_iz_out)
		np.save(self.cwd+"/"+self.podDirectory+'/sigmaS01_POD_iz_out', self.sigmaS01_POD_iz_out)
		
		print 'SVD S10_dof'
		S10_dof_iz_out = np.delete(self.S10_dof_full,self.stress_dofinside, axis=0)		
		self.S10_POD_iz_out,self.sigmaS10_POD_iz_out,self.WS10_POD_iz_out = np.linalg.svd(S10_dof_iz_out,0)
		np.save(self.cwd+"/"+self.podDirectory+'/S10_POD_iz_out', self.S10_POD_iz_out)
		np.save(self.cwd+"/"+self.podDirectory+'/sigmaS10_POD_iz_out', self.sigmaS10_POD_iz_out)
		time_end = tm.time()
		print "svd time : ", time_end-time_start, " s"
	


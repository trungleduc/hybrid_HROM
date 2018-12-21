

import scipy as sci
import numpy as np
from DataCreator import *
from PODCreator import *
from dolfin import *
from tools import *
import csv

class rid(data_creator):
	def __init__(self,meshInput,lzInput,numberModeInputForRID,ssListInput):
		data_creator.__init__(self,meshInput,lzInput,ssListInput)
		#create dir and files to save data
		self.set_savedir()
		# create reduced basis to build the RID
		self.create_dataRID(meshInput,lzInput,numberModeInputForRID,ssListInput  )	
		# create the list of coordinate to build the RID
		self.create_RID_coordinate(meshInput)
		self.create_RID_domain(meshInput,lzInput)
		self.create_RID_DOFs_list(meshInput)
		self.create_VPOD_over_RID(meshInput)
	# override load_data() of data_creator to load VPOD created by PODCreator
	def load_data(self):
		self.V_POD =np.load(self.cwd+"/"+self.podDirectory+'/V_POD.npy')
		self.S_POD=np.load(self.cwd+"/"+self.podDirectory+'/S_POD_out.npy')
		self.S1_POD=np.load(self.cwd+"/"+self.podDirectory+'/S1_POD_out.npy')
		self.S01_POD=np.load(self.cwd+"/"+self.podDirectory+'/S01_POD_out.npy')
		self.S10_POD=np.load(self.cwd+"/"+self.podDirectory+'/S10_POD_out.npy')
	def set_savedir(self):
		savedir = "RID"
		J0xml = File("J0domain.xml")
		self.file_subdomain = File(self.cwd+"/"+self.ridDirectory+"/subdomain.xml")
		self.RID = File(self.cwd+"/"+self.ridDirectory+"/RID.pvd")
		self.IZ_file = File(self.cwd+"/"+self.ridDirectory+"/IZ.pvd")
		self.file_submesh=File(self.cwd+"/"+self.ridDirectory+"/submesh.pvd")
	
	def create_dataRID(self,mesh,lz,numberModeInputForRID,ssListInput):		
		dataRID = pod(mesh,lz,numberModeInputForRID,ssListInput)

		self.V_POD_RID =dataRID.V_POD
		self.V_POD_out_full_RID =dataRID.V_POD_iz_out_full
		self.S_POD_RID=dataRID.S_POD_iz_out_full
		self.S1_POD_RID=dataRID.S1_POD_iz_out_full
		self.S01_POD_RID=dataRID.S01_POD_iz_out_full
		self.S10_POD_RID=dataRID.S10_POD_iz_out_full
		print np.shape(self.V_POD_out_full_RID)
	def create_RID_coordinate(self,mesh1):
		dofinside0 = np.asarray(self.IZ_dofinside)
		print'deim V_POD'
		P_dof, self.F_dof_out = deim(self.V_POD_out_full_RID)
		#self.F_dof=np.append(dofinside0,self.F_dof_out)
		self.F_dof = self.F_dof_out
		
		print'deim S_POD'
		PS_dof, FS_dof = deim(self.S_POD_RID)
		
		print'deim S1_POD'
		PS1_dof, FS1_dof = deim(self.S1_POD_RID)
		
		print'deim S01_POD'
		PS01_dof, FS01_dof = deim(self.S01_POD_RID)

		print'deim S10_POD'
		PS10_dof, FS10_dof = deim(self.S10_POD_RID)
		
		FStress_dof = list(set(FS_dof) | set(FS1_dof) | set(FS01_dof) | set(FS10_dof)) 
		mesh1.init(self.tdim - 1, self.tdim)
		mesh1.init(self.tdim - 2, self.tdim)
		V = VectorFunctionSpace(mesh1, 'CG', 1)
		dofmap = V.dofmap()
		dofs = dofmap.dofs()
		# Get coordinates as len(dofs) x gdim array
		dofs_x = dofmap.tabulate_all_coordinates(mesh1).reshape((-1, self.gdim))
		self.rid_coor=[]
		for dof in self.F_dof:
			self.rid_coor.append(dofs_x[dof])

		for dof1 in FStress_dof:
			self.rid_coor.append(dofs_x[dof1*2])  


		print "number of node used to build RID : ", np.shape(self.rid_coor)[0]
	
	def create_RID_domain(self,mesh1,lz):
		print "Loop over all cells to build RID ... "
		self.subdomains = CellFunction('size_t', mesh1, 0)
		for cell in cells(mesh1):
			for i in range(len(self.rid_coor)):
				if (cell.contains(Point(self.rid_coor[i]))):
					self.subdomains[cell] = 1
					if cell.midpoint()[0]>=0.9*lz:
						for vectex_cell in vertices(cell):
							for cell1 in cells(vectex_cell) :
								self.subdomains[cell1] = 1

				
		class IZO(SubDomain):
			def inside(self, x, on_boundary):
				return True  if (x[0]<1.05*lz) and (x[0]>=0.) else False

		izon = IZO()
		izon.mark(self.subdomains, 1) 
		self.file_subdomain<<self.subdomains
		self.RID<<self.subdomains
		
	def create_RID_DOFs_list(self,mesh1):
		V1 = VectorFunctionSpace(mesh1, 'CG', 1)
		dx =  Measure('dx', domain=mesh1, subdomain_data=self.subdomains)
		du, du_t = TrialFunction(V1), TestFunction(V1)
		u = Function(V1)
		A_test= assemble(inner(du, du_t)*dx(0))
		A_test1= assemble(inner(du, du_t)*dx(1))

		Amatrx_sp = petsc_csr(A_test)
		Amatrx = Amatrx_sp.diagonal()
		Amatrx1_sp = petsc_csr(A_test1)
		Amatrx1 = Amatrx1_sp.diagonal()

		self.F_dofinside=[]
		self.F_dofoutside=[]
		for i in range (np.shape(Amatrx)[0]):
			if Amatrx[i] == 0 :
				self.F_dofinside.append(i)
		for i in range (np.shape(Amatrx1)[0]):
			if Amatrx1[i] != 0 :
				self.F_dofoutside.append(i)
		print len(self.F_dofinside)
		print len(self.F_dofoutside)

	def create_VPOD_over_RID(self,mesh1):
		self.Nrdof=np.shape(self.F_dofoutside)[0]
		self.Ndof = np.shape(self.V_POD)[0]
		self.VR0_POD = np.zeros(shape=(self.Nrdof,np.shape(self.V_POD)[1]))
		self.ZtZ0=np.zeros(shape=(self.Nrdof,self.Nrdof))
		for i in range(self.Nrdof):
			self.VR0_POD[i,:]=self.V_POD[self.F_dofoutside[i],:]
			if self.F_dofoutside[i] in self.F_dofinside:
				self.ZtZ0[i,i]=1
		
		self.submesh1 = SubMesh(mesh1, self.subdomains, 1)
		print 'node', self.submesh1.num_vertices()
		print 'cells', self.submesh1.num_cells()
		self.file_submesh<<self.submesh1
		
		self.mydomains = CellFunction('size_t', self.submesh1)
		self.mydomains.set_all(0)
		dx_subdomain = Measure('dx', domain=mesh1, subdomain_data=self.mydomains)
		Vt = VectorFunctionSpace(self.submesh1, "Lagrange", 1)
		V = VectorFunctionSpace(mesh1, 'CG', 1)
		gsub_dim = self.submesh1.geometry().dim()
		submesh1_dof_coordinates = Vt.dofmap().tabulate_all_coordinates(self.submesh1).reshape(-1, gsub_dim)
		mesh1_dof_coordinates = V.dofmap().tabulate_all_coordinates(mesh1).reshape(-1, gsub_dim)
		
		mesh1_dof_index_coordinates0={}
		for index,coor in enumerate(mesh1_dof_coordinates):
			mesh1_dof_index_coordinates0.setdefault(coor[0],[]).append(index)
		
		mesh1_dof_index_coordinates1={}
		for index,coor in enumerate(mesh1_dof_coordinates):
			mesh1_dof_index_coordinates1.setdefault(coor[1],[]).append(index)	
		
		sub_to_glob_map = {}
		for bnd_dof_nr, bnd_dof_coords in enumerate(submesh1_dof_coordinates):
			corresponding_dofs = np.intersect1d(mesh1_dof_index_coordinates0[bnd_dof_coords[0]], mesh1_dof_index_coordinates1[bnd_dof_coords[1]])     
			if corresponding_dofs[0] not in sub_to_glob_map.values():
				sub_to_glob_map[bnd_dof_nr] = corresponding_dofs[0]
			else:
				sub_to_glob_map[bnd_dof_nr] = corresponding_dofs[1]
		#print sub_to_glob_map
		glob_to_sub_map = dict((v,k) for k,v in sub_to_glob_map.items())
		#print glob_to_sub_map
		self.VR_POD=np.zeros(shape=(np.shape(self.VR0_POD)))
		self.ZtZ=np.zeros(shape=(self.Nrdof,self.Nrdof))
		for i in range(self.Nrdof):
			ai=glob_to_sub_map[self.F_dofoutside[i]]
			self.VR_POD[ai]=self.VR0_POD[i]
			self.ZtZ[ai,ai]=self.ZtZ0[i,i]
		
		np.save(self.cwd+"/"+self.rpodDirectory+'/ZtZ', self.ZtZ)
		np.save(self.cwd+"/"+self.rpodDirectory+'/VR_POD', self.VR_POD)
		np.save(self.cwd+"/"+self.rpodDirectory+'/F_inside', self.F_dofinside)
		np.save(self.cwd+"/"+self.rpodDirectory+'/F_outside', self.F_dofoutside)

		w = csv.writer(open("glob_to_sub_map.csv", "w"))
		for key, val in glob_to_sub_map.items():
			w.writerow([key, val])


"""
ONLINE PHASE aerostruct
This program aims to use the POD-ROM to create a reduced basis able to be used in a real-time application.
It is applied to an aeroelastic problem.
Using version 3.0.6 of GMSH.

July 2018
@author: ochandre
"""

import numpy as np
from sklearn.externals import joblib
import functions_Offline as foff
from mesh_deformation_airbus import mesh_deformation_airbus
from VLM import VLM_study
from FEM import FEM_study
import timeit
import subprocess
import shutil
from transfert import transfert_matrix

##LOADING... (FROM OFFLINE PROCESS)
nSamples = np.load("../results/Offline/nSamples.npy")
param_data = np.load("../results/Offline/param_data.npy")
u_name = "../results/Offline/uV.npy"
g_name = "../results/Offline/gV.npy"
uV = np.load(u_name)
gV = np.load(g_name)

##ONLINE TREATMENT
#Parameters initialisation
h_skins = 0.02
h_ribs = 0.01
h_spars_le = 0.05
h_spars_te = 0.04
b = 55.
S = 480.

x_conf = np.array([h_skins,h_ribs,h_spars_le,h_spars_te,b,S])
pC = np.zeros((6))
pC[0] = param_data[5] = h_skins
pC[1] = param_data[6] = h_ribs
pC[2] = param_data[7] = h_spars_le
pC[3] = param_data[8] = h_spars_te
pC[4] = b
pC[5] = S
# 1) Kriging prediction of the interested point
tic = timeit.default_timer()
u_mp = np.zeros((1,nSamples))
u_vp = np.zeros((1,nSamples))
gamma_mp = np.zeros((1,nSamples))
gamma_vp = np.zeros((1,nSamples))
for i in range(nSamples):
    # *Loading Kriging data
    name_a = joblib.load("../results/Offline/GP_alpha_"+str(i)+".pkl")
    name_b = joblib.load("../results/Offline/GP_beta_"+str(i)+".pkl")
    # **Prediction
    u_mp[0,i], u_vp[0,i] = name_a.predict(x_conf.reshape((1,6)), eval_MSE=True)
    gamma_mp[0,i], gamma_vp[0,i] = name_b.predict(x_conf.reshape((1,6)), eval_MSE=True)
u_mean = 0
u_var2 = 0
gamma_mean = 0
gamma_var2 = 0
for i in range(nSamples):
    u_mean += np.dot(u_mp[0,i],uV[:,i])
    u_var2 += np.dot(u_vp[0,i]**2,uV[:,i]**2)
    gamma_mean += np.dot(gamma_mp[0,i],gV[:,i])
    gamma_var2 += np.dot(gamma_vp[0,i]**2,gV[:,i]**2)
# 2) Calculation of u_est and g_est
u_est = u_mean
u_est1 = u_mean+3.0*np.sqrt(u_var2)
u_est2 = u_mean-3.0*np.sqrt(u_var2)
g_est = gamma_mean
g_est1 = gamma_mean+3.0*np.sqrt(gamma_var2)
g_est2 = gamma_mean-3.0*np.sqrt(gamma_var2)
toc = timeit.default_timer()
print("ONLINE COMPUTATION TIME: "+str(toc-tic)+" s")
# 3) Publication of the results: u_est, g_est
a_new = mesh_deformation_airbus('../mesh/param_wing/VLM_mesh.msh','../mesh/param_wing/FEM_mesh.msh') 
a_new.new_wing_mesh(b,S)
vlm_mesh = '../mesh/param_wing/new_VLM.msh'
fem_mesh = '../mesh/param_wing/new_FEM.msh'
n_VLM_nodes, n_FEM_nodes, n_gamma_nodes = foff.calcule_nodes(pC)
# 3a) Computation of strains and stress
element_property,material,element_type = foff.create_dict(param_data)
my_fem = FEM_study(fem_mesh,element_type,element_property,material)
my_fem.read_mesh_file()
strain_dict, stress_dict = my_fem.get_strain_and_stress(u_est)
# 3b) u_est
my_fem.post_processing(u_est,"../results/Param_wing/u_est")
my_fem.post_processing_var1(u_est1,"../results/Param_wing/u_est1")
my_fem.post_processing_var2(u_est2,"../results/Param_wing/u_est2")
## 3c) Von Mises stress
Von_Mises_Stress = my_fem.get_Von_Mises(stress_dict)
my_fem.post_processing_Von_Mises(Von_Mises_Stress,'../results/param_wing/result_VM_iter')
print("Average Von Mises Stress = "+str(np.mean(Von_Mises_Stress))+" Pa")
# 3d) g_est
my_vlm = VLM_study(vlm_mesh)
my_vlm.post_processing_gamma('Param_wing/g_est',g_est)
my_vlm.post_processing_gamma_var1('Param_wing/g_est1',g_est1)
my_vlm.post_processing_gamma_var2('Param_wing/g_est2',g_est2)
# 4) Solver original
tic = timeit.default_timer()
#New wing
a = mesh_deformation_airbus('../mesh/param_wing/VLM_mesh.msh','../mesh/param_wing/fem_mesh.msh') 
a.new_wing_mesh(b = b, S = S)
#VLM meshes of the parametric wing
vlm_mesh_file =  '../mesh/param_wing/new_VLM.msh'
vlm_mesh_file_out =  '../mesh/param_wing/new_VLM_def.msh'
#Beam FEM mesh of the parametric wing
fem_mesh = '../mesh/param_wing/new_fem.msh'
##initialization
#VLM_study
alpha = 2.5
le = [400,700]
te = [200,500]
my_vlm = VLM_study(vlm_mesh_file,alpha,le,te,symmetry = True, v_inf = 250.72, rho = 0.3629)
#deformed mesh file which is the same as the non deformed one at initialization
shutil.copyfile(vlm_mesh_file,vlm_mesh_file_out)
#FEM_study
E = 70*1e9
nu = 0.3
element_property = {'tri':{'"skins"':{'h':h_skins},'"ribs"':{'h':h_ribs},
                           '"spars_le"':{'h':h_spars_le},'"spars_te"':{'h':h_spars_te}}}
material = {'tri':{'"skins"':{'E':E,'nu':nu},'"ribs"':{'E':E,'nu':nu},
                   '"spars_le"':{'E':E,'nu':nu},'"spars_te"':{'E':E,'nu':nu}}}
element_type = {'tri':'DKT'}
my_fem = FEM_study(fem_mesh,element_type,element_property,material)
#transfert
my_vlm.read_gmsh_mesh()
vlm_nodes_tot = my_vlm.nodes.copy()
vlm_nodes = vlm_nodes_tot[:,1:]
vlm_nodes_ind = vlm_nodes_tot[:,0]
elem_dict,element_tot = my_fem.read_mesh_file()
fem_nodes_tot = elem_dict['nodes'].copy()
#for the transfert we only uses the nodes that belongs to the skins
elm_skins = elem_dict['element_sets'][1]['elements']
node_skins = np.unique(elm_skins[:,3:6])
node_skins = np.array(node_skins,dtype = int)
fem_nodes = fem_nodes_tot[node_skins-1,1:]
fem_nodes_ind = node_skins
H = transfert_matrix(fem_nodes,vlm_nodes,function_type='thin_plate')
###fixed point
err = 1.0
tol = 1e-3
itermax = 250
n = 0
u_s_0 = np.ones((len(fem_nodes_tot)*6,))*1000.0
gamma_0 = np.ones((np.sum([my_vlm.surfaces[i]['points'].shape[0]for i in range(2)]),))*1000.0
while n<itermax and err>tol:
    #run the VLM analysis
    my_vlm = VLM_study(vlm_mesh_file_out,alpha, le, te, symmetry = True, v_inf = 250.72, rho = 0.3629)
    gamma = my_vlm.compute_circulation()
    speed,forces = my_vlm.compute_velocities_and_forces(gamma)
    my_vlm.post_processing_gamma_reel("Param_wing/g_real",gamma)
    #compute forces at nodes
    F_a = my_vlm.compute_force_at_nodes(vlm_nodes,forces)
    #transfert this forces to the structural beam model 
    F_s = np.dot(H.T,F_a)
    #rigidity matrix
    K = my_fem.assembling_K()
    #creating the rhs for the fem study
    rhs = np.zeros((len(fem_nodes_tot)*6,))
    for i in range(3):
        rhs[6*(fem_nodes_ind-1)+i] = F_s[:,i]
    my_fem.set_rhs(rhs)    
    #Boundary condition
    ind = fem_nodes[:,1]<=0.06
    nodes_sets = [fem_nodes_ind[ind]]
    dof = [[0,1,2,3,4,5]]
    my_fem.boundary_conditions(nodes_sets,dof)
    #solving
    u_s_tot = my_fem.solve()
    my_fem.post_processing_reel(u_s_tot,"../results/Param_wing/u_real")
    #Strain and Stress
    strain_tot,stress_tot = my_fem.get_strain_and_stress(u_s_tot)
    Von_Mises_Stress = my_fem.get_Von_Mises(stress_tot)
    my_fem.post_processing_Von_Mises_reel(Von_Mises_Stress,"../results/Param_wing/result_VM")
    print("Average Von Mises Stress (orig) = "+str(np.mean(Von_Mises_Stress))+" Pa")
    #interpolation of the displacement at skin nodes
    u_a = np.zeros((len(vlm_nodes)*6,))
    for i in range(6):
        u_a[0+i::6] = np.dot(H,u_s_tot[6*(fem_nodes_ind-1)+i])
    #vlm mesh deformation
    new_nodes = vlm_nodes_tot.copy()
    new_nodes[:,1] = new_nodes[:,1] + u_a[0::6]
    new_nodes[:,2] = new_nodes[:,2] + u_a[1::6]
    new_nodes[:,3] = new_nodes[:,3] + u_a[2::6]
    my_vlm.deformed_mesh_file(new_nodes,vlm_mesh_file_out)
    #compute relative error
    err = np.max([np.linalg.norm(u_s_0-u_s_tot)/np.linalg.norm(u_s_tot),np.linalg.norm(gamma_0-gamma)/np.linalg.norm(gamma)])
    u_s_0 = u_s_tot.copy()
    gamma_0 = gamma.copy()
    n = n+1
    print('--------------------'+str(n)+'---------------------')
toc = timeit.default_timer()
print("ORIGINAL CODE COMPUTATION TIME: "+str(toc-tic)+" s")
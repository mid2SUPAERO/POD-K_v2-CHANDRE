import numpy as np
from VLM import VLM_study
from FEM import FEM_study
from transfert import transfert_matrix
import shutil

class PARAM_WING_VLM_FEM():    
    def __init__(self,param_data = np.zeros(9)):
        self.alpha = param_data[0]
        self.v_inf = param_data[1]
        self.rho = param_data[2]
        self.E = param_data[3]
        self.nu = param_data[4]
        self.h_skins = param_data[5]
        self.h_ribs = param_data[6]
        self.h_spars_le = param_data[7]
        self.h_spars_te = param_data[8]
        self.le = [400,700]
        self.te = [200,500]

    def do_analysis(self,vlm_mesh_file,vlm_mesh_file_out,fem_mesh):
        #VLM_study
        my_vlm = VLM_study(vlm_mesh_file,self.alpha,self.le,self.te,self.v_inf,self.rho)
        #deformed mesh file which is the same as the non deformed one at initialization
        shutil.copyfile(vlm_mesh_file,vlm_mesh_file_out)
        #FEM_study
        element_property = {'tri':{'"skins"':{'h':self.h_skins},'"ribs"':{'h':self.h_ribs},
                                   '"spars_le"':{'h':self.h_spars_le},'"spars_te"':{'h':self.h_spars_te}}}
        material = {'tri':{'"skins"':{'E':self.E,'nu':self.nu},'"ribs"':{'E':self.E,'nu':self.nu},
                           '"spars_le"':{'E':self.E,'nu':self.nu},'"spars_te"':{'E':self.E,'nu':self.nu}}}
        element_type = {'tri':'DKT'}
        my_fem = FEM_study(fem_mesh,element_type,element_property,material)
        #Transfert
        my_vlm.read_gmsh_mesh()
        vlm_nodes_tot = my_vlm.nodes.copy()
        vlm_nodes = vlm_nodes_tot[:,1:]
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
        itermax = 100
        n = 0
        u_s_0 = np.ones((len(fem_nodes_tot)*6,))*1000.0
        gamma_0 = np.ones((np.sum([my_vlm.surfaces[i]['points'].shape[0]for i in range(2)]),))*1000.0
        while n<itermax and err>tol:
#            print "------aerostruct Iteration num.: ",n
            #run the VLM analysis
            my_vlm = VLM_study(vlm_mesh_file_out,self.alpha,self.le,self.te,self.v_inf,self.rho)
            gamma = my_vlm.compute_circulation()
            A = my_vlm.get_A()
            B = my_vlm.get_B()
            speed,forces = my_vlm.compute_velocities_and_forces(gamma)
            L,D = my_vlm.compute_L_D(forces)
            CL,CD = my_vlm.compute_CL_CD(L,D)
#            my_vlm.post_processing('param_wing/result_iter'+str(n),gamma,forces)
            #compute forces at nodes
            F_a = my_vlm.compute_force_at_nodes(vlm_nodes,forces)
            #transfert this forces to the structural beam model
            F_s = np.dot(H.T,F_a)
            #rigidity matrix
            my_fem.assembling_K()
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
#            my_fem.post_processing(u_s_tot,'../results/param_wing/result_struct_iter'+str(n))
#            #Strain and Stress
#            strain_tot,stress_tot = my_fem.get_strain_and_stress(u_s_tot)
#            my_fem.post_processing_strain_stress(strain_tot,stress_tot,'../results/param_wing/result_ss_iter'+str(n))
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
        return u_s_tot, gamma
    
    def get_nodes(self,fem_mesh,my_vlm):
        #FEM_study
        element_property = {'tri':{'"skins"':{'h':self.h_skins},'"ribs"':{'h':self.h_ribs},
                                   '"spars_le"':{'h':self.h_spars_le},'"spars_te"':{'h':self.h_spars_te}}}
        material = {'tri':{'"skins"':{'E':self.E,'nu':self.nu},'"ribs"':{'E':self.E,'nu':self.nu},
                           '"spars_le"':{'E':self.E,'nu':self.nu},'"spars_te"':{'E':self.E,'nu':self.nu}}}
        element_type = {'tri':'DKT'}
        my_fem = FEM_study(fem_mesh,element_type,element_property,material)
        #Transfert
        my_vlm.read_gmsh_mesh()
        vlm_nodes_tot = my_vlm.nodes.copy()
        vlm_nodes = vlm_nodes_tot[:,1:]
        elem_dict,element_tot = my_fem.read_mesh_file()
        fem_nodes_tot = elem_dict['nodes'].copy()
        #for the transfert we only uses the nodes that belongs to the skins
        elm_skins = elem_dict['element_sets'][1]['elements']
        node_skins = np.unique(elm_skins[:,3:6])
        node_skins = np.array(node_skins,dtype = int)
        fem_nodes = fem_nodes_tot[node_skins-1,1:]
        return vlm_nodes, fem_nodes
import numpy as np
"""
Implementation of triangular finite elements
@ Sylvain DUBREUIL, ONERA
"""
class DKT_element():
    def __init__(self,element_property,material):
        self.element_property = element_property
        self.material = material
        self.int_nodes = np.array([[1.0/6.0,1.0/6.0],
                        [2.0/3.0,1.0/6.0],
                        [1.0/6.0,2.0/3.0]])                 
        self.int_weights = np.ones((3,))*1.0/6.0
        self.xi_k = np.array([0.0,1.0,0.0])
        self.eta_k = np.array([0.0,0.0,1.0])
        
    def shape_function(self,xi,eta):
        Lambda = 1.0-xi-eta 
        N_1 = Lambda
        N_2 = xi
        N_3 = eta
        P_4 = 4.0*xi*Lambda
        P_5 = 4.0*xi*eta
        P_6 = 4.0*eta*Lambda
        return np.array([N_1,N_2,N_3,P_4,P_5,P_6])
        
    
    
    def glob_to_loc(self,nodes):
        #nodes coord are expressed in the global coordinate system
        #we first create a local elementary coordinate system
        u = nodes[1,:]-nodes[0,:]
        v_temp = nodes[2,:] - nodes[0,:]
        n = np.cross(u,v_temp)
        v = np.cross(n,u)
        
        u = u/np.linalg.norm(u)
        v = v/np.linalg.norm(v)
        n = n/np.linalg.norm(n)

        r_glob_to_loc = np.zeros((3,3))
        r_glob_to_loc[0,:] = u
        r_glob_to_loc[1,:] = v
        r_glob_to_loc[2,:] = n
        return r_glob_to_loc
        
    def compute_Jacobian(self,coord_loc,xi,eta):
        #Jacobian matrix, transformation from the reference element to the element in local coordinate system
        d_N = self.d_shape_function(xi,eta)
        J = np.zeros((2,2))
        J[0,0] = np.sum(d_N[0:3,0]*coord_loc[:,0])
        J[0,1] = np.sum(d_N[0:3,0]*coord_loc[:,1])
        J[1,0] = np.sum(d_N[0:3,1]*coord_loc[:,0])
        J[1,1] = np.sum(d_N[0:3,1]*coord_loc[:,1])
        
        det_J = J[0,0]*J[1,1]-J[0,1]*J[1,0]
        
        inv_J = np.zeros((2,2))
        inv_J[0,0] = J[1,1]
        inv_J[1,1] = J[0,0]
        inv_J[0,1] = -J[0,1]
        inv_J[1,0] = -J[1,0]
        inv_J = 1.0/det_J * inv_J
        
        return  J, inv_J,d_N, det_J
        
    def d_shape_function(self,xi,eta):
        N_1_xi = -1.0
        N_1_eta = -1.0
        N_2_xi = 1.0
        N_2_eta = 0.0
        N_3_xi = 0.0
        N_3_eta = 1.0
        P_4_xi = 4.0*(1.0-2.0*xi-eta)
        P_4_eta = -4.0*xi
        P_5_xi = 4.0*eta
        P_5_eta = 4.0*xi
        P_6_xi = -4.0*eta
        P_6_eta = 4.0*(1.0-xi-2.0*eta)
        return np.array([[N_1_xi,N_1_eta],[N_2_xi,N_2_eta],[N_3_xi,N_3_eta],[P_4_xi,P_4_eta],[P_5_xi,P_5_eta],[P_6_xi,P_6_eta]])
        
    def compute_B_m(self,inv_J,d_N):
        #membrane deformation w.r.t dof u and v
        B_m = np.zeros((3,6))
        for i in range(3):
            B_m[0,(2*i)] = inv_J[0,0]*d_N[i,0]+inv_J[0,1]*d_N[i,1]
            B_m[1,(2*i)+1] = inv_J[1,0]*d_N[i,0]+inv_J[1,1]*d_N[i,1]
            B_m[2,(2*i)] = inv_J[1,0]*d_N[i,0]+inv_J[1,1]*d_N[i,1]
            B_m[2,(2*i)+1] = inv_J[0,0]*d_N[i,0]+inv_J[0,1]*d_N[i,1]
        return B_m    

    def compute_B_f(self, inv_J,d_N,d_P,L,Cos,Sin,xi,eta):
        #bending deformation w.r.t dof w beta_x beta_y
        B_x_xi = np.zeros((9,))
        B_x_xi[0] = 6.*d_P[0,0]*Cos[0]/(4.*L[0])-6.*d_P[2,0]*Cos[2]/(4.*L[2])
        B_x_xi[1] = d_N[0,0]-3./4.*(d_P[0,0]*Cos[0]**2+d_P[2,0]*Cos[2]**2)
        B_x_xi[2] = -3./4.*(d_P[0,0]*Cos[0]*Sin[0]+d_P[2,0]*Cos[2]*Sin[2])
        B_x_xi[3] = 6.*d_P[1,0]*Cos[1]/(4.*L[1])-6.*d_P[0,0]*Cos[0]/(4.*L[0])
        B_x_xi[4] = d_N[1,0]-3./4.*(d_P[1,0]*Cos[1]**2+d_P[0,0]*Cos[0]**2)
        B_x_xi[5] = -3./4.*(d_P[1,0]*Cos[1]*Sin[1]+d_P[0,0]*Cos[0]*Sin[0])
        B_x_xi[6] = 6.*d_P[2,0]*Cos[2]/(4.*L[2])-6.*d_P[1,0]*Cos[1]/(4.*L[1])
        B_x_xi[7] = d_N[2,0]-3./4.*(d_P[2,0]*Cos[2]**2+d_P[1,0]*Cos[1]**2)
        B_x_xi[8] = -3./4.*(d_P[2,0]*Cos[2]*Sin[2]+d_P[1,0]*Cos[1]*Sin[1])

        B_x_eta = np.zeros((9,))
        B_x_eta[0] = 6.*d_P[0,1]*Cos[0]/(4.*L[0])-6.*d_P[2,1]*Cos[2]/(4.*L[2])
        B_x_eta[1] = d_N[0,1]-3./4.*(d_P[0,1]*Cos[0]**2+d_P[2,1]*Cos[2]**2)
        B_x_eta[2] = -3./4.*(d_P[0,1]*Cos[0]*Sin[0]+d_P[2,1]*Cos[2]*Sin[2])
        B_x_eta[3] = 6.*d_P[1,1]*Cos[1]/(4.*L[1])-6.*d_P[0,1]*Cos[0]/(4.*L[0])
        B_x_eta[4] = d_N[1,1]-3./4.*(d_P[1,1]*Cos[1]**2+d_P[0,1]*Cos[0]**2)
        B_x_eta[5] = -3./4.*(d_P[1,1]*Cos[1]*Sin[1]+d_P[0,1]*Cos[0]*Sin[0])
        B_x_eta[6] = 6.*d_P[2,1]*Cos[2]/(4.*L[2])-6.*d_P[1,1]*Cos[1]/(4.*L[1])
        B_x_eta[7] = d_N[2,1]-3./4.*(d_P[2,1]*Cos[2]**2+d_P[1,1]*Cos[1]**2)
        B_x_eta[8] = -3./4.*(d_P[2,1]*Cos[2]*Sin[2]+d_P[1,1]*Cos[1]*Sin[1])
        
        B_y_xi = np.zeros((9,))
        B_y_xi[0] = 6.*d_P[0,0]*Sin[0]/(4.*L[0])-6.*d_P[2,0]*Sin[2]/(4.*L[2])
        B_y_xi[1] = -3./4.*(d_P[0,0]*Cos[0]*Sin[0]+d_P[2,0]*Cos[2]*Sin[2])
        B_y_xi[2] = d_N[0,0]-3./4.*(d_P[0,0]*Sin[0]**2+d_P[2,0]*Sin[2]**2)
        B_y_xi[3] = 6.*d_P[1,0]*Sin[1]/(4.*L[1])-6.*d_P[0,0]*Sin[0]/(4.*L[0])
        B_y_xi[4] = -3./4.*(d_P[1,0]*Cos[1]*Sin[1]+d_P[0,0]*Cos[0]*Sin[0])
        B_y_xi[5] = d_N[1,0]-3./4.*(d_P[1,0]*Sin[1]**2+d_P[0,0]*Sin[0]**2)
        B_y_xi[6] = 6.*d_P[2,0]*Sin[2]/(4.*L[2])-6.*d_P[1,0]*Sin[1]/(4.*L[1])
        B_y_xi[7] = -3./4.*(d_P[2,0]*Cos[2]*Sin[2]+d_P[1,0]*Cos[1]*Sin[1])
        B_y_xi[8] = d_N[2,0]-3./4.*(d_P[2,0]*Sin[2]**2+d_P[1,0]*Sin[1]**2)
        
        B_y_eta = np.zeros((9,))
        B_y_eta[0] = 6.*d_P[0,1]*Sin[0]/(4.*L[0])-6.*d_P[2,1]*Sin[2]/(4.*L[2])
        B_y_eta[1] = -3./4.*(d_P[0,1]*Cos[0]*Sin[0]+d_P[2,1]*Cos[2]*Sin[2])
        B_y_eta[2] = d_N[0,1]-3./4.*(d_P[0,1]*Sin[0]**2+d_P[2,1]*Sin[2]**2)
        B_y_eta[3] = 6.*d_P[1,1]*Sin[1]/(4.*L[1])-6.*d_P[0,1]*Sin[0]/(4.*L[0])
        B_y_eta[4] = -3./4.*(d_P[1,1]*Cos[1]*Sin[1]+d_P[0,1]*Cos[0]*Sin[0])
        B_y_eta[5] = d_N[1,1]-3./4.*(d_P[1,1]*Sin[1]**2+d_P[0,1]*Sin[0]**2)
        B_y_eta[6] = 6.*d_P[2,1]*Sin[2]/(4.*L[2])-6.*d_P[1,1]*Sin[1]/(4.*L[1])
        B_y_eta[7] = -3./4.*(d_P[2,1]*Cos[2]*Sin[2]+d_P[1,1]*Cos[1]*Sin[1])
        B_y_eta[8] = d_N[2,1]-3./4.*(d_P[2,1]*Sin[2]**2+d_P[1,1]*Sin[1]**2)
        
        B_f = np.zeros((3,9))
        B_f[0,:] = inv_J[0,0]*B_x_xi+inv_J[0,1]*B_x_eta
        B_f[1,:] = inv_J[1,0]*B_y_xi+inv_J[1,1]*B_y_eta
        B_f[2,:] = inv_J[0,0]*B_y_xi+inv_J[0,1]*B_y_eta+inv_J[1,0]*B_x_xi+inv_J[1,1]*B_x_eta
        

        return B_f        
        
        
    def compute_K_elem(self,nodes):
        #constitutive law matrices
        E = self.material['E']
        nu = self.material['nu']
        h = self.element_property['h']
        H_temp = np.eye(3)
        H_temp[0,1] = H_temp[1,0] = nu
        H_temp[2,2] = (1.0-nu)/2.0
        H_m = E*h/(1.0-nu**2)*H_temp
        H_f = E*h**3/(12.0*(1.0-nu**2))*H_temp
        r_glob_to_loc = self.glob_to_loc(nodes)
        #nodes coordinates in the local coordinate system 
        coord_loc = np.dot(r_glob_to_loc,nodes.T).T
        #lenghts, sinus and cosinus
        lengths = []
        lengths.append(np.sqrt((coord_loc[1,0]-coord_loc[0,0])**2+(coord_loc[1,1]-coord_loc[0,1])**2))
        lengths.append(np.sqrt((coord_loc[2,0]-coord_loc[1,0])**2+(coord_loc[2,1]-coord_loc[1,1])**2))
        lengths.append(np.sqrt((coord_loc[0,0]-coord_loc[2,0])**2+(coord_loc[0,1]-coord_loc[2,1])**2))
        Cos = []
        Cos.append((coord_loc[1,0]-coord_loc[0,0])/lengths[0])
        Cos.append((coord_loc[2,0]-coord_loc[1,0])/lengths[1])
        Cos.append((coord_loc[0,0]-coord_loc[2,0])/lengths[2])
        Sin = []
        Sin.append((coord_loc[1,1]-coord_loc[0,1])/lengths[0])
        Sin.append((coord_loc[2,1]-coord_loc[1,1])/lengths[1])
        Sin.append((coord_loc[0,1]-coord_loc[2,1])/lengths[2])
        #Rigidity matrices
        K_m = np.zeros((6,6))
        K_f_temp = np.zeros((9,9))
        #Numerical integration
        k = 0
        for xi,eta in self.int_nodes:
            J, inv_J,d_N,det_J = self.compute_Jacobian(coord_loc,xi,eta)
            d_P = d_N[3:,:]
            B_m = self.compute_B_m(inv_J,d_N)
            K_m = K_m + self.int_weights[k]*(np.dot(B_m.T,np.dot(H_m,B_m)))*abs(det_J)
            B_f = self.compute_B_f(inv_J,d_N,d_P,lengths,Cos,Sin,xi,eta)
            K_f_temp = K_f_temp +self.int_weights[k]*(np.dot(B_f.T,np.dot(H_f,B_f)))*abs(det_J)
            k=k+1
        
        #Warning! K_f acts on the dof wi,beta_x,beta_y with beta_x = theta_y and beta_y = -theta_x
        #before assembling the total elementary matrix we transform K_f 
        K_f_temp_2 = np.zeros((9,9))
        for i in range(3):
            K_f_temp_2[:,i*3] = K_f_temp[:,i*3].copy()
            K_f_temp_2[:,(i*3)+1] = -K_f_temp[:,(i*3)+2].copy()
            K_f_temp_2[:,(i*3)+2] = K_f_temp[:,(i*3)+1].copy()
        K_f = np.zeros((9,9))
        for i in range(3):
            K_f[i*3,:] = K_f_temp_2[i*3,:].copy()
            K_f[(i*3)+1,:] = -K_f_temp_2[(i*3)+2,:].copy()
            K_f[(i*3)+2,:] = K_f_temp_2[(i*3)+1,:].copy()            
       #small rigidity for the theta_z dof
        ind = K_f.diagonal()!=0.0
        min_val = np.min(K_f.diagonal()[ind])
        epsilon = 1e-5*min_val 
        #total rigidity matrix (adding a fictious rotation about local z and renumbering the dof such as
        #U =[u_i,v_i,w_i,theta_x_i,theta_y_i,theta_z_i]
        K_e = np.zeros((18,18))
        for i in range(3):
            K_e[(i*6)+0:(i*6)+2,(i*6)+0:(i*6)+2] = K_m[(i*2)+0:(i*2)+2,(i*2)+0:(i*2)+2]
            K_e[(i*6)+2:(i*6)+5,(i*6)+2:(i*6)+5] = K_f[(i*3)+0:(i*3)+3,(i*3)+0:(i*3)+3] 
            #adding a fictious rigidity to the theta_z dof 
            K_e[(i*6)+5,(i*6)+5] = epsilon
            for j in range(2-i):                
                K_e[(i+j+1)*6+0:(i+j+1)*6+2,(i*6)+0:(i*6)+2] = K_m[(i+j+1)*2+0:(i+j+1)*2+2,(i*2)+0:(i*2)+2]
                K_e[(i*6)+0:(i*6)+2,(i+j+1)*6+0:(i+j+1)*6+2] = K_m[(i*2)+0:(i*2)+2,(i+j+1)*2+0:(i+j+1)*2+2]
                K_e[(i+j+1)*6+2:(i+j+1)*6+5,(i*6)+2:(i*6)+5] = K_f[(i+j+1)*3+0:(i+j+1)*3+3,(i*3)+0:(i*3)+3] 
                K_e[(i*6)+2:(i*6)+5,(i+j+1)*6+2:(i+j+1)*6+5] = K_f[(i*3)+0:(i*3)+3,(i+j+1)*3+0:(i+j+1)*3+3] 
        
        #K_e is expressed in the local coordinate system, we rotate it to the global one
        R = np.zeros((18,18))
        for i in range(6):
            R[(i*3):(i*3)+3,(i*3):(i*3)+3] = r_glob_to_loc
        K_e_g = np.dot(R.T,np.dot(K_e,R))
        
        return K_e_g
    
    def compute_strain_and_stress(self,nodes,U_elm,z = "max"):
        #1) remettre U_elm dans le repere de l'element et remettre en fonction de beta_x et beta_y 
        #2) Pour chaque point de Gauss calculer la matrice B_m (constante sur l'elm) et B_f (non constante sur l'elm)
        #3) Calcul de epsilon membrane et epsilon flexion aux points de gauss
        if z == "max":
            z = self.element_property['h']/2.0
        E = self.material['E']
        nu = self.material['nu']
        C = E/(1.0-nu**2)*np.array([[1.,nu,0.],[nu,1.,0.],[0.,0.,0.5*(1.-nu)]])
        r_glob_to_loc = self.glob_to_loc(nodes)
        #nodes coordinates in the local coordinate system 
        coord_loc = np.dot(r_glob_to_loc,nodes.T).T
        #lenghts, sinus and cosinus
        lengths = []
        lengths.append(np.sqrt((coord_loc[1,0]-coord_loc[0,0])**2+(coord_loc[1,1]-coord_loc[0,1])**2))
        lengths.append(np.sqrt((coord_loc[2,0]-coord_loc[1,0])**2+(coord_loc[2,1]-coord_loc[1,1])**2))
        lengths.append(np.sqrt((coord_loc[0,0]-coord_loc[2,0])**2+(coord_loc[0,1]-coord_loc[2,1])**2))
        Cos = []
        Cos.append((coord_loc[1,0]-coord_loc[0,0])/lengths[0])
        Cos.append((coord_loc[2,0]-coord_loc[1,0])/lengths[1])
        Cos.append((coord_loc[0,0]-coord_loc[2,0])/lengths[2])
        Sin = []
        Sin.append((coord_loc[1,1]-coord_loc[0,1])/lengths[0])
        Sin.append((coord_loc[2,1]-coord_loc[1,1])/lengths[1])
        Sin.append((coord_loc[0,1]-coord_loc[2,1])/lengths[2])
        
        #1) rotation of U_elm
        R = np.zeros((18,18))
        for i in range(6):
            R[(i*3):(i*3)+3,(i*3):(i*3)+3] = r_glob_to_loc
        U_elm_loc = np.dot(R,U_elm).T
        #1-1) Warning! B matrices are formulated with rotation beta_x = theta_y and beta_y = -theta_x
        for i in range(3):
            theta_x = U_elm_loc[(6*i)+3].copy()
            theta_y = U_elm_loc[(6*i)+4].copy()
            U_elm_loc[(6*i)+3] = theta_y
            U_elm_loc[(6*i)+4] = -theta_x
        
        #2) compute the B matrices at each gauss point and calculate the strain        
        k = 0
        strain = {}
        stress = {}
        strain={}
        stress={}
        for xi,eta in self.int_nodes:
            strain[k] = {}
            stress[k] = {}
            J, inv_J,d_N,det_J = self.compute_Jacobian(coord_loc,xi,eta)
            d_P = d_N[3:,:]
            #membrane strain w.r.t dof u and v (does not depend on the gauss point)
            #(e_xx,e_xy,2e_xy) = \sum_{i=1}^3 B_mk U_k (U_k = (u_k, v_k))
            B_m = self.compute_B_m(inv_J,d_N)
            U_uv = U_elm_loc[[0,1,6,7,12,13]]
            strain_m = np.dot(B_m,U_uv)
            epsilon_m = np.zeros((3,3))
            epsilon_m[0,0] = strain_m[0]
            epsilon_m[1,1] = strain_m[1]
            epsilon_m[0,1] = epsilon_m[1,0] = strain_m[2]            
            #bending strain w.r.t dof w beta_x and beta_y (depend on the gauss point)
            #(k_xx,k_xy,2 k_xy) = \sum_{i=1}^3 B_fk U_k (U_k = (w_k, beta_x_k,beta_y_k))
            B_f = self.compute_B_f(inv_J,d_N,d_P,lengths,Cos,Sin,xi,eta)
            U_w_beta = U_elm_loc[[2,3,4,8,9,10,14,15,16]]
            strain_b = np.dot(B_f,U_w_beta)
            epsilon_b = np.zeros((3,3))
            epsilon_b[0,0] = strain_b[0]
            epsilon_b[1,1] = strain_b[1]
            epsilon_b[0,1] = epsilon_b[1,0] = strain_b[2]
            #total strain 
            strain_tot = strain_m+z*strain_b
            epsilon_tot = np.zeros((3,3))
            epsilon_tot[0,0] = strain_tot[0]
            epsilon_tot[1,1] = strain_tot[1]
            epsilon_tot[0,1] = epsilon_tot[1,0] = strain_tot[2]
            #3) compute stress
            stress_m = np.dot(C,strain_m)
            sigma_m = np.zeros((3,3))
            sigma_m[0,0] = stress_m[0]
            sigma_m[1,1] = stress_m[1]
            sigma_m[0,1] = sigma_m[1,0] = stress_m[2]             
            stress_b = z*np.dot(C,strain_b)
            sigma_b = np.zeros((3,3))
            sigma_b[0,0] = stress_b[0]
            sigma_b[1,1] = stress_b[1]
            sigma_b[0,1] = sigma_b[1,0] = stress_b[2]             
            stress_tot = stress_m+stress_b
            sigma_tot = np.zeros((3,3))
            sigma_tot[0,0] = stress_tot[0]
            sigma_tot[1,1] = stress_tot[1]
            sigma_tot[0,1] = sigma_tot[1,0] = stress_tot[2] 
            #4) coordinate of the gauss point in global coordinate system
            N = self.shape_function(xi,eta)[0:3]
            x = np.dot(N,coord_loc[:,0])
            y = np.dot(N,coord_loc[:,1])
            gauss_point_coord = np.array([x,y,coord_loc[:,2].mean()]) 
            gauss_point_coord = np.dot(r_glob_to_loc.T,gauss_point_coord)
            #strain and stress are expressed in the element coordinate system, 
            #we want to express it in the global one
            epsilon_m_global = np.dot(r_glob_to_loc.T,np.dot(epsilon_m,r_glob_to_loc))            
            epsilon_b_global = np.dot(r_glob_to_loc.T,np.dot(epsilon_b,r_glob_to_loc))
            epsilon_tot_global = np.dot(r_glob_to_loc.T,np.dot(epsilon_tot,r_glob_to_loc))
            
            sigma_m_global = np.dot(r_glob_to_loc.T,np.dot(sigma_m,r_glob_to_loc))            
            sigma_b_global = np.dot(r_glob_to_loc.T,np.dot(sigma_b,r_glob_to_loc))
            sigma_tot_global = np.dot(r_glob_to_loc.T,np.dot(sigma_tot,r_glob_to_loc))
            
            strain[k]['coord'] = gauss_point_coord
            strain[k]['element_coord_system']={}
            strain[k]['element_coord_system']['membrane'] = epsilon_m
            strain[k]['element_coord_system']['bending'] = epsilon_b
            strain[k]['element_coord_system']['total'] = epsilon_tot
            strain[k]['global_coord_system']={}
            strain[k]['global_coord_system']['membrane'] = epsilon_m_global
            strain[k]['global_coord_system']['bending'] = epsilon_b_global
            strain[k]['global_coord_system']['total'] = epsilon_tot_global            
            
            
            stress[k]['coord'] = gauss_point_coord
            stress[k]['element_coord_system']={}
            stress[k]['element_coord_system']['membrane'] = sigma_m
            stress[k]['element_coord_system']['bending'] = sigma_b
            stress[k]['element_coord_system']['total'] = sigma_tot
            stress[k]['global_coord_system']={}
            stress[k]['global_coord_system']['membrane'] = sigma_m_global
            stress[k]['global_coord_system']['bending'] = sigma_b_global
            stress[k]['global_coord_system']['total'] = sigma_tot_global           
            k=k+1
        return strain, stress    

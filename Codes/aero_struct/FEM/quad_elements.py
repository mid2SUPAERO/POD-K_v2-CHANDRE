import numpy as np
"""
Implementation of quadrilateral finite elements
"""

class Q4_element():
    def __init__(self,element_property,material):
        self.element_property = element_property
        self.material = material
        self.int_nodes = np.ones((4,2))*1.0/np.sqrt(3.0)
        ind = np.array([[False,False],
                        [False,True],
                        [True,False],
                        [True,True]])
        self.int_nodes[ind] = -1.0 * self.int_nodes[ind]                 
        self.int_weights = np.ones((4,))
        self.xi_k = np.array([-1.0,1.0,1.0,-1.0])
        self.eta_k = np.array([-1.0,-1.0,1.0,1.0])
        
    def shape_function(self,xi,eta):
        N_1 = 0.25*(1.0-xi)*(1.0-eta)
        N_2 = 0.25*(1.0+xi)*(1.0-eta)
        N_3 = 0.25*(1.0+xi)*(1.0+eta)
        N_4 = 0.25*(1.0-xi)*(1.0+eta)
        return np.array([N_1,N_2,N_3,N_4])
        
    def d_shape_function(self,xi,eta):
        N_1_xi = -0.25*(1.0-eta)
        N_1_eta = -0.25*(1.0-xi)
        N_2_xi = 0.25*(1.0-eta)
        N_2_eta = -0.25*(1.0+xi)
        N_3_xi = 0.25*(1.0+eta)
        N_3_eta = 0.25*(1.0+xi)
        N_4_xi = -0.25*(1.0+eta)
        N_4_eta = 0.25*(1.0-xi)
        return np.array([[N_1_xi,N_1_eta],[N_2_xi,N_2_eta],[N_3_xi,N_3_eta],[N_4_xi,N_4_eta]])
    
    def glob_to_loc(self,nodes):
        #nodes coord are expressed in the global coordinate system
        #we first create a local elementary coordinate system
        u = nodes[1,:]-nodes[0,:]
        v_temp = nodes[3,:] - nodes[0,:]
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
        J[0,0] = np.sum(d_N[:,0]*coord_loc[:,0])
        J[0,1] = np.sum(d_N[:,0]*coord_loc[:,1])
        J[1,0] = np.sum(d_N[:,1]*coord_loc[:,0])
        J[1,1] = np.sum(d_N[:,1]*coord_loc[:,1])
        
        det_J = J[0,0]*J[1,1]-J[0,1]*J[1,0]
        
        inv_J = np.zeros((2,2))
        inv_J[0,0] = J[1,1]
        inv_J[1,1] = J[0,0]
        inv_J[0,1] = -J[0,1]
        inv_J[1,0] = -J[1,0]
        inv_J = 1.0/det_J * inv_J
        
        return  J, inv_J,d_N, det_J
    
    def compute_B_m(self,inv_J,d_N):
        #membrane deformation w.r.t dof u and v
        B_m = np.zeros((3,8))
        for i in range(4):
            B_m[0,(2*i)] = inv_J[0,0]*d_N[i,0]+inv_J[0,1]*d_N[i,1]
            B_m[1,(2*i)+1] = inv_J[1,0]*d_N[i,0]+inv_J[1,1]*d_N[i,1]
            B_m[2,(2*i)] = inv_J[1,0]*d_N[i,0]+inv_J[1,1]*d_N[i,1]
            B_m[2,(2*i)+1] = inv_J[0,0]*d_N[i,0]+inv_J[0,1]*d_N[i,1]
        return B_m
    
    def compute_B_c(self,coord_loc,inv_J,d_N,N):
        #distorsion transverse
        #Interpolation lineaire et distorsion verifie faiblement sur les bords
        B_loc = np.zeros((2,12))
        for i in range(4):
            #compute the jacobian matrix at node point
            J_i, _,_,_ = self.compute_Jacobian(coord_loc,self.xi_k[i],self.eta_k[i])
            B_loc[0,(3*i)] = d_N[i,0]
            B_loc[0,(3*i)+1] = self.xi_k[i]*J_i[0,0]*d_N[i,0]
            B_loc[0,(3*i)+2] = self.xi_k[i]*J_i[0,1]*d_N[i,0]
            B_loc[1,(3*i)] = d_N[i,1]
            B_loc[1,(3*i)+1] = self.eta_k[i]*J_i[1,0]*d_N[i,1]
            B_loc[1,(3*i)+2] = self.eta_k[i]*J_i[1,1]*d_N[i,1]   
        B_c = np.dot(inv_J,B_loc)
        #Interpolation lineaire et sous integration
#        B_c = np.zeros((2,12))
#        for i in range(4):
#            B_c[0,(3*i)] = inv_J[0,0]*d_N[i,0]+inv_J[0,1]*d_N[i,1]
#            B_c[0,(3*i)+1] = N[i]
#            B_c[1,(3*i)] = inv_J[1,0]*d_N[i,0]+inv_J[1,1]*d_N[i,1]
#            B_c[1,(3*i)+2] = N[i]   
        return B_c
    
    def compute_B_f(self,inv_J,d_N):
        #flexion
        B_f = np.zeros((3,12))
        for i in range(4):
            B_f[0,(3*i)+1] = inv_J[0,0]*d_N[i,0]+inv_J[0,1]*d_N[i,1]
            B_f[1,(3*i)+2] = inv_J[1,0]*d_N[i,0]+inv_J[1,1]*d_N[i,1]
            B_f[2,(3*i)+1] = inv_J[1,0]*d_N[i,0]+inv_J[1,1]*d_N[i,1]
            B_f[2,(3*i)+2] = inv_J[0,0]*d_N[i,0]+inv_J[0,1]*d_N[i,1]
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
        H_c = 5.0/6.0*E*h/(2.0*(1.0+nu))*np.eye(2)
        r_glob_to_loc = self.glob_to_loc(nodes)
        #nodes coordinates in the local coordinate system 
        coord_loc = np.dot(r_glob_to_loc,nodes.T).T
        #Rigidity matrices
        K_m = np.zeros((8,8))
        K_f_temp = np.zeros((12,12))
        K_c_temp = np.zeros((12,12))
        #Numerical integration
        k = 0
        for xi,eta in self.int_nodes:
            J, inv_J,d_N,det_J = self.compute_Jacobian(coord_loc,xi,eta)
            N = self.shape_function(xi,eta)
            B_m = self.compute_B_m(inv_J,d_N)
            K_m = K_m + self.int_weights[k]*(np.dot(B_m.T,np.dot(H_m,B_m)))*abs(det_J)
            B_f = self.compute_B_f(inv_J,d_N)
            K_f_temp = K_f_temp +self.int_weights[k]*(np.dot(B_f.T,np.dot(H_f,B_f)))*abs(det_J)
            #interpolation lineaire et distorsion verifie faiblement sur les bords
            B_c = self.compute_B_c(coord_loc,inv_J,d_N,N)
            K_c_temp = K_c_temp + self.int_weights[k]*(np.dot(B_c.T,np.dot(H_c,B_c)))*abs(det_J)
            k=k+1
        #interpolation lineaire et sous integration    
#        xi = 0.0
#        eta = 0.0    
#        J, inv_J,d_N,det_J = self.compute_Jacobian(coord_loc,xi,eta)
#        N = self.shape_function(xi,eta)
#        B_c = self.compute_B_c(coord_loc,inv_J,d_N,N)
#        K_c_temp = K_c_temp + 1.0*(np.dot(B_c.T,np.dot(H_c,B_c)))*abs(det_J)

        
        #Warning! K_f and Kc act on the dof wi,beta_x,beta_y with beta_x = theta_y and beta_y = -theta_x
        #before assembling the total elementary matrix we transform K_f and K_c
        K_f_temp_2 = np.zeros((12,12))
        K_c_temp_2 = np.zeros((12,12))
        for i in range(4):
            K_f_temp_2[:,i*3] = K_f_temp[:,i*3].copy()
            K_c_temp_2[:,i*3] = K_c_temp[:,i*3].copy()
            K_f_temp_2[:,(i*3)+1] = -K_f_temp[:,(i*3)+2].copy()
            K_f_temp_2[:,(i*3)+2] = K_f_temp[:,(i*3)+1].copy()
            K_c_temp_2[:,(i*3)+1] = -K_c_temp[:,(i*3)+2].copy()
            K_c_temp_2[:,(i*3)+2] = K_c_temp[:,(i*3)+1].copy()
        K_f = np.zeros((12,12))
        K_c = np.zeros((12,12))
        for i in range(4):
            K_f[i*3,:] = K_f_temp_2[:,i*3].copy()
            K_c[i*3,:] = K_c_temp_2[:,i*3].copy()
            K_f[(i*3)+1,:] = -K_f_temp_2[:,(i*3)+2].copy()
            K_f[(i*3)+2,:] = K_f_temp_2[:,(i*3)+1].copy()
            K_c[(i*3)+1,:] = -K_c_temp_2[:,(i*3)+2].copy()
            K_c[(i*3)+2,:] = K_c_temp_2[:,(i*3)+1].copy()    
        #small rigidity for the theta_z dof
        ind = K_f.diagonal()!=0.0
        min_val = np.min(K_f.diagonal()[ind])
        epsilon = 1e-5*min_val 
        #total rigidity matrix (adding a fictious rotation about local z and renumbering the dof such as
        #U =[u_i,v_i,w_i,theta_x_i,theta_y_i,theta_z_i]
        K_e = np.zeros((24,24))
        for i in range(4):
            K_e[(i*6)+0:(i*6)+2,(i*6)+0:(i*6)+2] = K_m[(i*2)+0:(i*2)+2,(i*2)+0:(i*2)+2]
            K_e[(i*6)+2:(i*6)+5,(i*6)+2:(i*6)+5] = K_f[(i*3)+0:(i*3)+3,(i*3)+0:(i*3)+3] + K_c[(i*3)+0:(i*3)+3,(i*3)+0:(i*3)+3]            
            #adding a fictious rigidity to the theta_z dof 
            K_e[(i*6)+5,(i*6)+5] = epsilon
            for j in range(3-i):                
                K_e[(i+j+1)*6+0:(i+j+1)*6+2,(i*6)+0:(i*6)+2] = K_m[(i+j+1)*2+0:(i+j+1)*2+2,(i*2)+0:(i*2)+2]
                K_e[(i*6)+0:(i*6)+2,(i+j+1)*6+0:(i+j+1)*6+2] = K_m[(i*2)+0:(i*2)+2,(i+j+1)*2+0:(i+j+1)*2+2]
                K_e[(i+j+1)*6+2:(i+j+1)*6+5,(i*6)+2:(i*6)+5] = K_f[(i+j+1)*3+0:(i+j+1)*3+3,(i*3)+0:(i*3)+3] + K_c[(i+j+1)*3+0:(i+j+1)*3+3,(i*3)+0:(i*3)+3] 
                K_e[(i*6)+2:(i*6)+5,(i+j+1)*6+2:(i+j+1)*6+5] = K_f[(i*3)+0:(i*3)+3,(i+j+1)*3+0:(i+j+1)*3+3] + K_c[(i*3)+0:(i*3)+3,(i+j+1)*3+0:(i+j+1)*3+3] 
        
        #K_e is expressed in the local coordinate system, we rotate it to the global one
        R = np.zeros((24,24))
        for i in range(8):
            R[(i*3):(i*3)+3,(i*3):(i*3)+3] = r_glob_to_loc
        K_e_g = np.dot(R.T,np.dot(K_e,R))
        
        return K_e_g

        
        
             
            
                                    
            
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
        
        
        
        
        
        
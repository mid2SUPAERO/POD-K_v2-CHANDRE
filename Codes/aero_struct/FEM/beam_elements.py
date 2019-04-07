import numpy as np
"""
Implementation of beam finite elements
"""
class beam_element():
    def __init__(self,element_property,material,model = 'Euler'):
        self.element_property = element_property
        self.material = material
        self.model = model        
        
    def compute_K_elem(self,nodes):

        coord_N_1 = nodes[0,:]
        coord_N_2 = nodes[1,:]       
        #element length
        L = np.linalg.norm(coord_N_2-coord_N_1)
        #property of the element
        E = self.material['E']        
        G = self.material['G']
        C = self.element_property['C']
        S = self.element_property['S']
        I_y = self.element_property['I_y']
        I_z = self.element_property['I_z']
        if self.model == 'Timoshenko':
            k_y = self.element_property['k_y']
            k_z = self.element_property['k_z']
       
        #elementary rigidity matrix
        K_elem = np.zeros((12,12))
        #Traction-compresion
        cste = E*S/L 
        K_elem[0,0] = K_elem[6,6] = cste
        K_elem[0,6] = K_elem[6,0] = -cste
        #Torsion
        cste = G*C/L
        K_elem[3,3] = K_elem[9,9] = cste
        K_elem[3,9] = K_elem[9,3] = -cste
        #bending (xOz)
        if self.model == 'Timoshenko':
            phi_y = 12.0*E*I_y/(k_z*S*G*L**2)
        else:
            phi_y = 0.0
        cste = 12.0*E*I_y/(L**3*(1.+phi_y))
        K_elem[2,2] = K_elem[8,8] = cste
        K_elem[4,4] = K_elem[10,10] = cste*1.0/12.0*(4.0+phi_y)*L**2
        K_elem[2,4] = K_elem[4,2] = K_elem[2,10] = K_elem[10,2] = cste*(-L/2.)
        K_elem[2,8] = K_elem[8,2] = -cste
        K_elem[4,8] = K_elem[8,4] = K_elem[8,10] = K_elem[10,8] = cste*L/2.
        K_elem[4,10] = K_elem[10,4] = cste*1.0/12.0*(2.0-phi_y)*L**2
        #bending (xOy)
        if self.model == 'Timoshenko':
            phi_z = 12.0*E*I_z/(k_y*S*G*L**2)
        else:
            phi_z = 0.0
        cste = 12.0*E*I_z/(L**3*(1.+phi_z))
        K_elem[1,1] = K_elem[7,7] = cste
        K_elem[5,5] = K_elem[11,11] = cste*1.0/12.0*(4.0+phi_z)*L**2
        K_elem[1,5] = K_elem[5,1] = K_elem[1,11] = K_elem[11,1] = cste*(L/2.)
        K_elem[1,7] = K_elem[7,1] = -cste
        K_elem[5,7] = K_elem[7,5] = K_elem[7,11] = K_elem[11,7] = cste*(-L/2.)
        K_elem[5,11] = K_elem[11,5] = cste*1.0/12.0*(2.0-phi_z)*L**2
        #rotation of the elementary rigidity matrix into the global coordinate system
        #local coordinate system expressed in the global one
        u = coord_N_2-coord_N_1
        z = np.array([0.0,0.0,1.0])
        v= np.zeros((3,))
        #is u || z ?
        theta = self.element_property['theta'] 
        if np.allclose(np.linalg.norm(np.cross(u,z)),1e-9,1e-9) == False:
            czu = np.cross(z,u)
            n = czu/np.linalg.norm(czu)            
            v[0] = n[0]*np.cos(theta)-n[1]*u[2]*np.sin(theta)
            v[1] = n[0]*u[2]*np.sin(theta)+n[1]*np.cos(theta)
            v[2] = np.sin(theta)*np.sqrt(u[0]**2+u[1]**1)
        else:
            v[0] = np.sin(theta)
            v[1] = np.cos(theta)
            v[2] = 0.0            
        v = v/np.linalg.norm(v)
        w = np.cross(u,v)
        w = w/np.linalg.norm(w)        
        #rotation matrix
        r = np.zeros((3,3))
        r[0,:] = u
        r[1,:] = v
        r[2,:] = w
        R = np.zeros((12,12))
        R[0:3,0:3] = r
        R[3:6,3:6] = r
        R[6:9,6:9] = r
        R[9:,9:] = r
        #rotation of K_elem
        K_e_g = np.dot(R.T,np.dot(K_elem,R))
        
        return K_e_g     
        
        
        
        
        
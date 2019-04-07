#mesh deformation tools

import numpy as np
import scipy.interpolate as inter
from define_geo import airbus_wing

class mesh_deformation_airbus():
    def __init__(self,VLM_mesh,fem_mesh):
        wing = airbus_wing()
        span_1, span_2 =  wing.compute_span()
        c_0, c_1, c_2, c_3 = wing.compute_cord()
        e_1,e_2,e_3 = wing.compute_ep()  
        phi, _  = wing.compute_phi()
        d = wing.compute_diedre()
        self.span_1 = span_1
        self.span_2 = span_2
        self.c_0 = c_0 
        self.c_1 = c_1
        self.c_2 = c_2
        self.c_3 = c_3
        self.phi = phi
        self.d = d
        self.VLM_mesh = VLM_mesh
        self.fem_mesh = fem_mesh
        self.y = [span_1]
        self.control_sections = self.defined_control_points()
        self.new_nodes = self.nodes.copy()
    def defined_control_points(self):
        #y = list of y coord where the control section should be defined
        n = len(self.y)
        f = open(self.VLM_mesh,'r')
        lines = f.readlines()
        f.close()
        #nodes
        ind_start_node = lines.index('$Nodes\r\n')
        ind_stop_node = lines.index('$EndNodes\r\n')
        n_nodes = int(lines[ind_start_node+1])
        nodes = np.zeros((n_nodes,4)) #nodes[:,0] = nodes number, nodes[:,1:]= nodes coordinates (x,y,z)
        i=0
        for line in lines[ind_start_node+2:ind_stop_node]:
            split_line=line.split()
            nodes[i,0]=int(split_line[0])
            nodes[i,1]=float(split_line[1])
            nodes[i,2]=float(split_line[2])
            nodes[i,3]=float(split_line[3])
            i=i+1 
        #sort nodes per sections
        ind = nodes[:,2].argsort()    
        nodes = nodes[ind,:]    
        d = np.append(True,np.diff(nodes[:,2]))
        tol = 1e-8
        y_sections = nodes[:,2][d>tol]
        n_sections = len(y_sections)
        n_chord = n_nodes/n_sections
        #find the n control sections
        control_sections = np.zeros((n+2,n_chord,4))
        control_sections[0,:,:] = nodes[0:n_chord]
        control_sections[n+1,:,:] = nodes[-n_chord::]      
        for i in range(n):
            #section close to y_i
            ind = abs(y_sections-self.y[i]).argmin()
            control_sections[i+1,:,:] = nodes[ind*n_chord:(ind+1)*n_chord]
        #control sections 1/4 chord points
        self.control_points = np.zeros((n+2,3))
        for i in range(n+2):
            A = control_sections[i,control_sections[i,:,1].argmin(),1::]
            B = control_sections[i,control_sections[i,:,1].argmax(),1::]
            vect = B-A
            if i == 0:
                self.control_points[i,:] = A+0.25*vect/np.linalg.norm(vect)*self.c_0
            else:
                self.control_points[i,:] = A+0.25*vect
        self.nodes = nodes
        self.y_sections = y_sections
        self.n_chord = n_chord
        return control_sections                         
        
    def apply_span(self,span_1, span_2):
        #span = value of the new span
        #translation of each section, value of the translation is assumed linear
        A1 = self.control_points[0] - [0.25*self.c_0,0,0]
        A2 = self.control_points[0] + [self.c_1-0.25*self.c_0,0,0]
        A3 = self.control_points[1] - [0.25*self.c_2,0,0]
        A4 = self.control_points[1] + [0.75*self.c_2,0,0]
        A5 = self.control_points[2] - [0.25*self.c_3,0,0]
        A6 = self.control_points[2] + [0.75*self.c_3,0,0]
        phi1 = np.arctan((A3[0]-A1[0])/(A3[1]-A1[1]))
        phi2 = np.arctan((A4[0]-A2[0])/(A4[1]-A2[1]))
        phi3 = np.arctan((A5[0]-A3[0])/(A5[1]-A3[1]))
        phi4 = np.arctan((A6[0]-A4[0])/(A6[1]-A4[1]))
        actual_span_1 = self.span_1
        delta_span_1 = span_1-actual_span_1
        f_delta_span_1 = inter.interp1d([self.control_points[0,1],self.control_points[1,1]],[0.0,delta_span_1],fill_value="extrapolate")
        actual_span_2 = self.span_2 
        delta_span_2 = span_2-actual_span_2
        f_delta_span_2 = inter.interp1d([self.control_points[1,1],self.control_points[2,1]],[0.0,delta_span_2],fill_value="extrapolate")
        #y translation of the control points
        delta_y_1 = f_delta_span_1(self.control_points[1,1])
        self.control_points[1,1] = self.control_points[1,1] + delta_y_1
        self.control_points[1,0] = self.control_points[1,0] + np.tan(self.phi)*delta_y_1
        delta_y_2 = f_delta_span_2(self.control_points[2,1])
        self.control_points[2,1] = self.control_points[2,1] + delta_y_1 + delta_y_2
        self.control_points[2,0] = self.control_points[2,0] + np.tan(self.phi)*(delta_y_1 + delta_y_2)
        for i in range(len(self.y_sections)-1):
            if self.y_sections[i+1] <= actual_span_1:
                xmin = self.new_nodes[(i+1)*self.n_chord:(i+2)*self.n_chord,1].min()
                xmax = self.new_nodes[(i+1)*self.n_chord:(i+2)*self.n_chord,1].max()
                f_phi = inter.interp1d([xmin,xmax],[phi1,phi2],fill_value="extrapolate")
                delta_span_i = f_delta_span_1(self.y_sections[i+1])
                self.new_nodes[(i+1)*self.n_chord:(i+2)*self.n_chord,2] = self.new_nodes[(i+1)*self.n_chord:(i+2)*self.n_chord,2] + delta_span_i
                for j in range(self.n_chord):
                    phi_j = f_phi(self.new_nodes[(i+1)*self.n_chord+j,1])
                    self.new_nodes[(i+1)*self.n_chord+j,1] = self.new_nodes[(i+1)*self.n_chord+j,1] + np.tan(phi_j)*delta_span_i
            else:                
                xmin = self.new_nodes[(i+1)*self.n_chord:(i+2)*self.n_chord,1].min()
                xmax = self.new_nodes[(i+1)*self.n_chord:(i+2)*self.n_chord,1].max()
                f_phi = inter.interp1d([xmin,xmax],[phi3,phi4],fill_value="extrapolate")
                delta_span_i = delta_y_1 + f_delta_span_2(self.y_sections[i+1])
                self.new_nodes[(i+1)*self.n_chord:(i+2)*self.n_chord,2] = self.new_nodes[(i+1)*self.n_chord:(i+2)*self.n_chord,2] + delta_span_i
                for j in range(self.n_chord):
                    phi_j = f_phi(self.new_nodes[(i+1)*self.n_chord+j,1])
                    self.new_nodes[(i+1)*self.n_chord+j,1] = self.new_nodes[(i+1)*self.n_chord+j,1] + np.tan(phi_j)*delta_span_i                     
            self.y_sections[i+1] = self.y_sections[i+1] + delta_span_i
        self.span_1 = span_1
        self.span_2 = span_2
        return self.new_nodes
    
    def apply_sweep(self, sweep):  
        #x translation of the control points
        delta_x = np.zeros((self.control_points.shape[0],))
        for i in range(np.shape(self.control_points)[0]-1):
            delta_x[i+1] = np.tan(sweep-self.phi)*self.control_points[i+1,1]
            self.control_points[i+1,0] = self.control_points[i+1,0] + delta_x[i+1]
        #linear interpolation
        f_delta_x_i = inter.interp1d(self.control_points[:,1],delta_x,fill_value="extrapolate")  
        for i in range(len(self.y_sections)-1):
            delta_x_i = f_delta_x_i(self.new_nodes[(i+1)*self.n_chord,2])
            self.new_nodes[(i+1)*self.n_chord:(i+2)*self.n_chord,1] = \
            self.new_nodes[(i+1)*self.n_chord:(i+2)*self.n_chord,1] + delta_x_i
        self.phi = sweep
        return self.new_nodes
    
    def apply_chord(self,c_0,c_1,c_2,c_3):
        #linear interpolation of the chord
        f_chord_primaire = inter.interp1d(self.control_points[:,1],[c_0,c_2,c_3],fill_value="extrapolate")
        f_chord_initiale = inter.interp1d(self.control_points[:,1],[self.c_0,self.c_2,self.c_3],fill_value="extrapolate")
        f_chord_x_i = inter.interp1d(self.control_points[:,1],[c_1,c_2,c_3],fill_value="extrapolate")
        for i in range(len(self.y_sections)):
            ind_x = self.new_nodes[i*self.n_chord:(i+1)*self.n_chord,1].argsort()
            self.new_nodes[i*self.n_chord:(i+1)*self.n_chord,:] = self.new_nodes[i*self.n_chord:(i+1)*self.n_chord,:][ind_x]
            A = self.new_nodes[i*self.n_chord,1:]
            B = self.new_nodes[(i+1)*self.n_chord-1,1:]
            #coordinate of the point at 0.25% of the actual chord
            vect = B-A
            vect_normed = vect/np.linalg.norm(vect)
            chord_i = f_chord_x_i(A[1])
            chord_prim_i = f_chord_primaire(A[1])
            chord_init_i = f_chord_initiale(A[1])
            
            C = A + 0.25 * vect_normed * chord_init_i                
            new_A = C-0.25*chord_prim_i*vect_normed
            new_B = C+(chord_i-0.25*chord_prim_i)*vect_normed
            delta_vect = (new_B-new_A)/self.n_chord
            for j in range(self.n_chord):
                self.new_nodes[i*self.n_chord:(i+1)*self.n_chord,:][j,1:] = new_A+j*delta_vect
        return self.new_nodes        
    
    def apply_diedre(self,d):
        #linear interpolation of the diedre
        delta_d = d - self.d
        for i in range(len(self.y_sections)-1):
            delta_z = np.tan(delta_d) * self.y_sections[i]
            self.new_nodes[(i+1)*self.n_chord:(i+2)*self.n_chord,3] = self.new_nodes[(i+1)*self.n_chord:(i+2)*self.n_chord,3] + delta_z
        return self.new_nodes     
     
    def new_wing_mesh(self,b,S,phi = 30.*np.pi/180.,diedre = 1.5*np.pi/180.,BF = 5.96,Mach = 0.85) :
        new_wing = airbus_wing(b,S,phi,diedre,BF,Mach)
        span_1, span_2 =  new_wing.compute_span()
        c_0, c_1, c_2, c_3 = new_wing.compute_cord()
        e_1,e_2,e_3 = new_wing.compute_ep()
        d = new_wing.compute_diedre() 
        phi_1, phi_2  = new_wing.compute_phi()
        self.apply_span(span_1,span_2)
        self.apply_chord(c_0,c_1,c_2,c_3)
        self.apply_sweep(phi_1)
        self.apply_diedre(d)
        self.apply_to_struct(self.fem_mesh)
        self.write_new_VLM_mesh_file('../mesh/param_wing/new_VLM.msh')   
        self.write_new_FEM_mesh_file('../mesh/param_wing/new_FEM.msh')
        
    
    def apply_to_struct(self,FEM_mesh_file):
        self.FEM_mesh_file = FEM_mesh_file
        ind = self.new_nodes[:,0].argsort()
        self.new_nodes = self.new_nodes[ind]
        ind = self.nodes[:,0].argsort()
        self.nodes = self.nodes[ind]        
        delta_VLM_nodes = self.new_nodes[:,1:]-self.nodes[:,1:]
        f = open(FEM_mesh_file,'r')
        lines = f.readlines()
        f.close()
        #nodes
        ind_start_node = lines.index('$Nodes\r\n')
        ind_stop_node = lines.index('$EndNodes\r\n')
        n_nodes = int(lines[ind_start_node+1])
        nodes = np.zeros((n_nodes,4)) #nodes[:,0] = nodes number, nodes[:,1:]= nodes coordinates (x,y,z)
        i=0
        for line in lines[ind_start_node+2:ind_stop_node]:
            split_line=line.split()
            nodes[i,0]=int(split_line[0])
            nodes[i,1]=float(split_line[1])
            nodes[i,2]=float(split_line[2])
            nodes[i,3]=float(split_line[3])
            i=i+1
        self.FEM_nodes = nodes
        ind = self.FEM_nodes[:,0].argsort()
        self.FEM_nodes= self.FEM_nodes[ind]
        #interpolation by RBF
        t_x = inter.Rbf(self.nodes[:,1],self.nodes[:,2],self.nodes[:,3],delta_VLM_nodes[:,0])
        t_y = inter.Rbf(self.nodes[:,1],self.nodes[:,2],self.nodes[:,3],delta_VLM_nodes[:,1])
        t_z = inter.Rbf(self.nodes[:,1],self.nodes[:,2],self.nodes[:,3],delta_VLM_nodes[:,2])
        t_x_fem = t_x(self.FEM_nodes[:,1],self.FEM_nodes[:,2],self.FEM_nodes[:,3])
        t_y_fem = t_y(self.FEM_nodes[:,1],self.FEM_nodes[:,2],self.FEM_nodes[:,3])
        t_z_fem = t_z(self.FEM_nodes[:,1],self.FEM_nodes[:,2],self.FEM_nodes[:,3])
        new_FEM_nodes = self.FEM_nodes.copy()
        self.new_FEM_nodes = new_FEM_nodes
        self.new_FEM_nodes[:,1] = self.FEM_nodes[:,1] + t_x_fem
        self.new_FEM_nodes[:,2] = self.FEM_nodes[:,2] + t_y_fem
        self.new_FEM_nodes[:,3] = self.FEM_nodes[:,3] + t_z_fem
        return self.new_FEM_nodes
        
    def write_new_VLM_mesh_file(self,new_mesh_file):
        n_nodes = self.nodes.shape[0]
        new_nodes = self.new_nodes
        f = open(self.VLM_mesh,'r')
        lines = f.readlines()
        f.close()
        ind_start_node = lines.index('$Nodes\r\n')
        ind_stop_node = lines.index('$EndNodes\r\n')
        f1 = open(new_mesh_file,'w')
        for line in lines[0:ind_start_node+2]:
            f1.write(line)
            
        for i in range(n_nodes):
            f1.write(str(int(new_nodes[i,0]))+' '+str(new_nodes[i,1])+' '+str(new_nodes[i,2])+' '+str(new_nodes[i,3])+'\n')
                
        for line in lines[ind_stop_node:]:
            f1.write(line)
        f1.close()
        
        return
        
    def write_new_FEM_mesh_file(self,new_mesh_file):
        n_nodes = self.FEM_nodes.shape[0]
        new_nodes = self.new_FEM_nodes
        f = open(self.FEM_mesh_file,'r')
        lines = f.readlines()
        f.close()
        ind_start_node = lines.index('$Nodes\r\n')
        ind_stop_node = lines.index('$EndNodes\r\n')
        f1 = open(new_mesh_file,'w')
        for line in lines[0:ind_start_node+2]:
            f1.write(line)
            
        for i in range(n_nodes):
            f1.write(str(int(new_nodes[i,0]))+' '+str(new_nodes[i,1])+' '+str(new_nodes[i,2])+' '+str(new_nodes[i,3])+'\n')
                
        for line in lines[ind_stop_node:]:
            f1.write(line)
        f1.close()
        
        return   

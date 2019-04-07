import numpy as np
import scipy.linalg as linalg
import scipy.sparse as spa
import scipy.sparse.linalg as spalinalg
from elements import finite_element
"""
Main class of the python FEM
@ Sylvain DUBREUIL, ONERA
"""
class FEM_study():
    """
    Construction:\n
    Inputs:\
    mesh_file: a msh mesh file\n
    element_type: dict whith key element type and asociated element formulation ex:element_type['beam']="Euler"\n
    availlable element_type are : \n
    element_type['beam']= "Euler" or "Timoshenko"\n
    element_type['tri']= "DKT"\n
    element_type['quad']= "Q4gamma" or "DKQ"\n
    """
    def __init__(self,mesh_file,element_type,element_property,material,mode = 'full'):
        self.mesh_file = mesh_file
        self.element_type = element_type
        self.element_property = element_property
        self.material = material
        self.mode = mode
    """
    Function that reads the mesh file and returns mesh dict\n
    Inputs:\n
    Outputs:\n
    element_dict: dict with key "nodes" and "elements"\n
    element_dict["nodes"] = array (n_nodes,4), number and coord of each nodes\n
    element_dict["elements"] = dict with keys "element_type"\n
    element_dict["element_type"] = array (n_elem,n_vertex+1), number and connectivity of each element
    element_dict["element_sets"] = dict with keys set_number
    element_dict["element_sets"][set_number] = dict with keys "name" and "elements"
    element_dict["element_sets"][set_number]["name"] = string, name of the set
    element_dict["element_sets"][set_number]["elements"] = array (n_elem in the set,n_vertex+1)
    """
    def read_mesh_file(self):
        f = open(self.mesh_file,'r')
        lines=f.readlines()
        #element sets
        try:
            ind_p_n = lines.index('$PhysicalNames\r\n')
            element_sets = {}
            n_sets = int(lines[ind_p_n+1].split()[0])
            for i in range(n_sets):
                element_sets[int(lines[ind_p_n+2+i].split()[1])] = {"name":lines[ind_p_n+2+i].split()[2],"elements" : []}
              
        except:
            pass
        
        
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
        #elements, any type
        ind_start_element = lines.index('$Elements\r\n')
        ind_stop_element = lines.index('$EndElements\r\n')
        n_elements = int(lines[ind_start_element+1])
        elements_tot = np.zeros((n_elements,7)) # elements_tot[:,0] = elements number, elements_tot[:,1] = elements type, elements_tot[:,2] = element_set, elements_tot[:,3:] = nodes number 
        i=0
        for line in lines[ind_start_element+2:ind_stop_element]:
            split_line=line.split()
            elements_tot[i,0]=int(split_line[0])
            elements_tot[i,1]=int(split_line[1])
            elements_tot[i,2]=int(split_line[3])
            if elements_tot[i,1] == 15:
                elements_tot[i,3]=int(split_line[5])
            elif elements_tot[i,1] == 1:
                elements_tot[i,3]=int(split_line[5])
                elements_tot[i,4]=int(split_line[6])
            elif elements_tot[i,1] == 2:
                elements_tot[i,3]=int(split_line[5])
                elements_tot[i,4]=int(split_line[6])
                elements_tot[i,5]=int(split_line[7])
            elif elements_tot[i,1] == 3:
                elements_tot[i,3]=int(split_line[5])
                elements_tot[i,4]=int(split_line[6])
                elements_tot[i,5]=int(split_line[7])
                elements_tot[i,6]=int(split_line[8])
            else:
                print "element type not implemented yet, element number = ",str(elements_tot[i,0])
            i=i+1
        #filling the element sets    
        for key in element_sets.iterkeys():
            element_sets[key]["elements"] = elements_tot[elements_tot[:,2]==key]
 
        #beam elements
        beam_elements_ind = elements_tot[:,1] == 1
        beam_elements = elements_tot[beam_elements_ind,:]
        n_beam = beam_elements.shape[0]
        beam_elements_temp = np.zeros((n_beam,3))
        beam_elements_temp[:,0] = beam_elements[:,0]
        beam_elements_temp[:,1:] = beam_elements[:,3:5]
        #triangular elements
        tri_elements_ind = elements_tot[:,1] == 2
        tri_elements = elements_tot[tri_elements_ind,:]
        n_tri = tri_elements.shape[0]
        tri_elements_temp = np.zeros((n_tri,4))
        tri_elements_temp[:,0] = tri_elements[:,0]
        tri_elements_temp[:,1:] = tri_elements[:,3:6]
        #quadrilateral elements
        quad_elements_ind = elements_tot[:,1] == 3
        quad_elements = elements_tot[quad_elements_ind,:]
        n_quad = quad_elements.shape[0]
        quad_elements_temp = np.zeros((n_quad,5))
        quad_elements_temp[:,0] = quad_elements[:,0]
        quad_elements_temp[:,1:] = quad_elements[:,3:7]
        
        element_dict = {'nodes':0,'elements':{}}
        element_dict['nodes'] = nodes
        self.nodes = nodes

                
        elem_nodes = []
        if n_beam>0:
            element_dict['elements']['beam'] = beam_elements_temp
            for elm in beam_elements_temp:
                if (elm[1] in elem_nodes) == False:
                    elem_nodes.append(elm[1])
                if (elm[2] in elem_nodes) == False:
                    elem_nodes.append(elm[2])    
        if n_tri>0:
            element_dict['elements']['tri'] = tri_elements_temp
            for elm in tri_elements_temp:
                if (elm[1] in elem_nodes) == False:
                    elem_nodes.append(elm[1])
                if (elm[2] in elem_nodes) == False:
                    elem_nodes.append(elm[2]) 
                if (elm[3] in elem_nodes) == False:
                    elem_nodes.append(elm[3])
                    
        if n_quad>0:
            element_dict['elements']['quad'] = quad_elements_temp
            for elm in quad_elements_temp:
                if (elm[1] in elem_nodes) == False:
                    elem_nodes.append(elm[1])
                if (elm[2] in elem_nodes) == False:
                    elem_nodes.append(elm[2]) 
                if (elm[3] in elem_nodes) == False:
                    elem_nodes.append(elm[3])
                if (elm[4] in elem_nodes) == False:
                    elem_nodes.append(elm[4])
        self.elem_nodes = elem_nodes
        element_dict["element_sets"] = element_sets            
        # Check if each nodes is used as a element vertex if not warning and keep the fake nodes in a list  
        self.fake_nodes = []
        for node in self.nodes:
            if (node[0] in elem_nodes) == False:
                self.fake_nodes.append(node[0])
        self.elements_tot = elements_tot 
        self.element_dict = element_dict
        return element_dict,elements_tot
        
        
    """
    Assembling the rigidity matrix
    """        
    def assembling_K(self):
        #read the mesh file
        element_dict, elements_tot = self.read_mesh_file()
        nodes_tot = element_dict['nodes']
        #initialization of the rigidity matrix
        #we assume that nodes are numbered from 1 to n_nodes 
        if self.mode == 'full':
            K = np.zeros((int(np.max(element_dict['nodes'][:,0])*6),int(np.max(element_dict['nodes'][:,0])*6)))
        elif self.mode == 'sparse':
            row = []
            col = []
            data = []
        #create reference element for each type of element
        reference_elements = {}
        for key_1 in element_dict['elements'].iterkeys():
            reference_elements[key_1] = {}
            for value in element_dict['element_sets'].itervalues():
                set_name = value['name']
                reference_elements[key_1][set_name] = finite_element(self.element_type[key_1],self.element_property[key_1][set_name],self.material[key_1][set_name])       
        #loop over the element
        elm_types = {1:"beam",2:"tri",3:"quad"}
        for elm in elements_tot:
            elm_set = element_dict['element_sets'][elm[2]]['name'] 
            elm_type = elm_types[elm[1]]
            if elm_type == "quad":
                n_nodes_elm = 4
            elif elm_type == "tri":
                n_nodes_elm =3
            elif elm_type == "beam":
                n_nodes_elm =2
            #for elm in elms:
            #nodes_elm = np.array(elm[1:],dtype=int)
            nodes_elm = np.array(elm[3:3+n_nodes_elm],dtype=int)    
            nodes_coord = np.zeros((len(nodes_elm),3))
            i = 0
            for node in nodes_elm:
                ind = node == nodes_tot[:,0]
                nodes_coord[i,:] = nodes_tot[ind][0,1:]
                i = i+1
            #Compute the element rigidity matrix  
            K_elem = reference_elements[elm_type][elm_set].compute_K(nodes_coord)
            #assembling
            if self.mode == 'full':
                for i in range(n_nodes_elm):
                    for j in range(n_nodes_elm):
                        K[6*(nodes_elm[i]-1):6*(nodes_elm[i]),6*(nodes_elm[j]-1):6*(nodes_elm[j])] = \
                        K[6*(nodes_elm[i]-1):6*(nodes_elm[i]),6*(nodes_elm[j]-1):6*(nodes_elm[j])] + \
                        K_elem[6*i:6*(i+1),6*j:6*(j+1)]
            elif self.mode == 'sparse':
                for i in range(n_nodes_elm*6):
                    for j in range(n_nodes_elm*6):
                        if K_elem[i,j] != 0.0:
                            data.extend([K_elem[i,j]])
                            ind_line = i/6
                            offset_line = i%6
                            row.extend([(nodes_elm[ind_line]-1)*6+offset_line])                            
                            ind_col = j/6
                            offset_col = j%6
                            col.extend([(nodes_elm[ind_col]-1)*6+offset_col])
        if self.mode == "full":                    
            self.K = K                
            return
        elif self.mode == "sparse":
            self.data = np.array(data)
            self.row = np.array(row)
            self.col = np.array(col)
            return
    """
    Apply boundary condition
    """
    def boundary_conditions(self,nodes_set,l_dof):
        i = 0    
        for nodes in nodes_set:
            for node in nodes:
                ind_node = np.argwhere(self.nodes[:,0]==node)[0,0] 
                if self.mode == 'full':
                    for dof in l_dof[i]:
                        self.K[:,6*ind_node+dof] = 0.0
                        self.K[6*ind_node+dof,:] = 0.0
                        self.K[6*ind_node+dof,6*ind_node+dof] = 1.0
                        self.rhs[6*ind_node+dof] = 0.0
                elif self.mode == 'sparse':
                    for dof in l_dof[i]:
                        ind_row = self.row == 6*ind_node+dof
                        self.data[ind_row] = 0.0
                        ind_col = self.col == 6*ind_node+dof
                        self.data[ind_col] = 0.0
                        ind_diag = ind_col & ind_row
                        self.data[ind_diag] = 1.0
                        self.rhs[6*ind_node+dof] = 0.0
            i = i + 1
        return 
    """
    Set RHS
    """    
    def set_rhs(self,rhs):
        #check that the dimension of the given rhs is correct
        if rhs.shape[0] != 6*len(self.nodes):
            print ('Warning! Dimension of RHS is incorrect')
        else:
            self.rhs = rhs
        return      
    
    """
    Solve KU=F
    """            
    def solve(self):
        if self.mode == 'full':
            U = linalg.solve(self.K,self.rhs)
        elif self.mode == 'sparse':
            K_coo = spa.coo_matrix((self.data, (self.row, self.col)), shape=(6*len(self.nodes), 6*len(self.nodes)))
            K_csc = K_coo.tocsc()
            U = spalinalg.spsolve(K_csc,self.rhs)
        return U            
        
    """
    Compute strain and stress
    """    
    def get_strain_and_stress(self,U):
        nodes_tot = self.element_dict['nodes']
        #loop over the element of the model 
        #create reference element for each type of element
        reference_elements = {}
        for key_1 in self.element_dict['elements'].iterkeys():
            reference_elements[key_1] = {}
            for value in self.element_dict['element_sets'].itervalues():
                set_name = value['name']
                reference_elements[key_1][set_name] = finite_element(self.element_type[key_1],self.element_property[key_1][set_name],self.material[key_1][set_name])       
        #loop over the element
        elm_types = {1:"beam",2:"tri",3:"quad"}
        strain_dict = {}
        stress_dict = {}
        for elm in self.elements_tot:
            elm_set = self.element_dict['element_sets'][elm[2]]['name'] 
            elm_type = elm_types[elm[1]]
            if elm_type == "quad":
                n_nodes_elm = 4
            elif elm_type == "tri":
                n_nodes_elm = 3
            elif elm_type == "beam":
                n_nodes_elm = 2
            nodes_elm = np.array(elm[3:3+n_nodes_elm],dtype=int)    
            nodes_coord = np.zeros((len(nodes_elm),3))
            i = 0
            for node in nodes_elm:
                ind = node == nodes_tot[:,0]
                nodes_coord[i,:] = nodes_tot[ind][0,1:]
                i = i+1
            #get the corresponding nodal solution
            U_elm = np.zeros((6*n_nodes_elm,))
            i = 0
            for node in nodes_elm:
                U_elm[i*6:(i+1)*6] = U[(node-1)*6:node*6] 
                i = i+1
            #Compute the element strain and stress at gauss points  
            strain,stress = reference_elements[elm_type][elm_set].compute_strain_and_stress(nodes_coord,U_elm)
            strain_dict[elm[0]] = strain
            stress_dict[elm[0]] = stress
        return strain_dict, stress_dict    
    
    """
    Compute Von Mises Stress
    """
    def get_Von_Mises(self,stress):
        Sigma_VM = np.zeros((len(self.elements_tot),))
        for i in range(len(self.elements_tot)):
            sigma_tot_0 = stress[i+1][0]['global_coord_system']['total']
            sigma_tot_1 = stress[i+1][1]['global_coord_system']['total']
            sigma_tot_2 = stress[i+1][2]['global_coord_system']['total'] 
            Sigma_tot = 1.0/3.0 * (sigma_tot_0+sigma_tot_1+sigma_tot_2)
            Sigma_VM[i] = np.sqrt(Sigma_tot[0,0]**2 - Sigma_tot[0,0]*Sigma_tot[1,1]+ Sigma_tot[1,1]**2 + 3*Sigma_tot[0,1]**2)
        return Sigma_VM
    
    """
    Displacement Post processing file for gmsh
    """    
    def post_processing(self,U,file_name):
       #copying mesh file
        with open(self.mesh_file) as f:
            with open(file_name+".msh", "w") as f1:
                for line in f:
                        f1.write(line)
        f.close()
        f1.close()
        f1 = open(file_name+".msh",'a')
        #vector displacement field at nodes
        f1.write("\r\n")
        f1.write("$NodeData\r\n")
        f1.write("1\r\n")
        f1.write('"Displacement (mean)"\r\n')
        f1.write("1\r\n")
        f1.write("0.0\r\n")
        f1.write("3\r\n")
        f1.write("0\r\n")
        f1.write("3\r\n")
        f1.write(str(int(len(self.nodes)))+"\r\n")
        for i in range(len(self.nodes)):
            f1.write(str(i+1)+" "+str(U[6*i])+" "+str(U[6*i+1])+" "+str(U[6*i+2])+"\r\n")
        f1.write("$EndNodeData")    
        return
    def post_processing_reel(self,U,file_name):
       #copying mesh file
        with open(self.mesh_file) as f:
            with open(file_name+".msh", "w") as f1:
                for line in f:
                        f1.write(line)
        f.close()
        f1.close()
        f1 = open(file_name+".msh",'a')
        #vector displacement field at nodes
        f1.write("\r\n")
        f1.write("$NodeData\r\n")
        f1.write("1\r\n")
        f1.write('"Displacement (orig)"\r\n')
        f1.write("1\r\n")
        f1.write("0.0\r\n")
        f1.write("3\r\n")
        f1.write("0\r\n")
        f1.write("3\r\n")
        f1.write(str(int(len(self.nodes)))+"\r\n")
        for i in range(len(self.nodes)):
            f1.write(str(i+1)+" "+str(U[6*i])+" "+str(U[6*i+1])+" "+str(U[6*i+2])+"\r\n")
        f1.write("$EndNodeData")    
        return
    def post_processing_var1(self,U,file_name):
       #copying mesh file
        with open(self.mesh_file) as f:
            with open(file_name+".msh", "w") as f1:
                for line in f:
                        f1.write(line)
        f.close()
        f1.close()
        f1 = open(file_name+".msh",'a')
        #vector displacement field at nodes
        f1.write("\r\n")
        f1.write("$NodeData\r\n")
        f1.write("1\n")
        f1.write('"Displacement (mean+3*var^2)"\r\n')
        f1.write("1\r\n")
        f1.write("0.0\r\n")
        f1.write("3\r\n")
        f1.write("0\r\n")
        f1.write("3\r\n")
        f1.write(str(int(len(self.nodes)))+"\r\n")
        for i in range(len(self.nodes)):
            f1.write(str(i+1)+" "+str(U[6*i])+" "+str(U[6*i+1])+" "+str(U[6*i+2])+"\r\n")
        f1.write("$EndNodeData")    
        return
    def post_processing_var2(self,U,file_name):
       #copying mesh file
        with open(self.mesh_file) as f:
            with open(file_name+".msh", "w") as f1:
                for line in f:
                        f1.write(line)
        f.close()
        f1.close()
        f1 = open(file_name+".msh",'a')
        #vector displacement field at nodes
        f1.write("\r\n")
        f1.write("$NodeData\r\n")
        f1.write("1\r\n")
        f1.write('"Displacement (mean-3*var^2)"\r\n')
        f1.write("1\r\n")
        f1.write("0.0\r\n")
        f1.write("3\r\n")
        f1.write("0\r\n")
        f1.write("3\r\n")
        f1.write(str(int(len(self.nodes)))+"\r\n")
        for i in range(len(self.nodes)):
            f1.write(str(i+1)+" "+str(U[6*i])+" "+str(U[6*i+1])+" "+str(U[6*i+2])+"\r\n")
        f1.write("$EndNodeData")    
        return

    """
    Strain and Stress Post processing file for gmsh
    """    
    def post_processing_strain_stress(self,strain,stress,file_name):
        #copying mesh file
        with open(self.mesh_file) as f:
            with open(file_name+"_mean_strain.msh", "w") as f1:
                for line in f:
                        f1.write(line)
        f.close()
        f1.close()
        f1 = open(file_name+"_mean_strain.msh",'a')
        #tensor at elements (mean of the gauss point)
        f1.write("\r\n")
        f1.write("$ElementData\r\n")
        f1.write("1\r\n")
        f1.write('"Strain tensor in global coordinate system"\r\n')
        f1.write("1\r\n")
        f1.write("0.0\r\n")
        f1.write("3\r\n")
        f1.write("0\r\n")
        f1.write("9\r\n")
        f1.write(str(int(len(self.elements_tot)))+"\r\n")
        for i in range(len(self.elements_tot)):
            epsilon_tot_0 = strain[i+1][0]['global_coord_system']['total']
            epsilon_tot_1 = strain[i+1][1]['global_coord_system']['total']
            epsilon_tot_2 = strain[i+1][2]['global_coord_system']['total'] 
            Epsilon_tot = 1.0/3.0 * (epsilon_tot_0+epsilon_tot_1+epsilon_tot_2)
            f1.write(str(i+1)+" "+str(Epsilon_tot[0,0])+" "+str(Epsilon_tot[0,1])+" "+str(Epsilon_tot[0,2])\
            +" "+str(Epsilon_tot[1,0])+" "+str(Epsilon_tot[1,1])+" "+str(Epsilon_tot[1,2])\
            +" "+str(Epsilon_tot[2,0])+" "+str(Epsilon_tot[2,1])+" "+str(Epsilon_tot[2,2])+"\r\n")
        f1.write("$EndElementData\r\n")
        f1.close()
        
        #copying mesh file
        with open(self.mesh_file) as f:
            with open(file_name+"_mean_stress.msh", "w") as f1:
                for line in f:
                        f1.write(line)
        f.close()
        f1.close()
        f1 = open(file_name+"_mean_stress.msh",'a')
        #tensor at elements (mean of the gauss point)
        f1.write("\r\n")
        f1.write("$ElementData\r\n")
        f1.write("1\r\n")
        f1.write('"Stress tensor in global coordinate system"\r\n')
        f1.write("1\r\n")
        f1.write("0.0\r\n")
        f1.write("3\r\n")
        f1.write("0\r\n")
        f1.write("9\r\n")
        f1.write(str(int(len(self.elements_tot)))+"\r\n")
        for i in range(len(self.elements_tot)):
            sigma_tot_0 = stress[i+1][0]['global_coord_system']['total']
            sigma_tot_1 = stress[i+1][1]['global_coord_system']['total']
            sigma_tot_2 = stress[i+1][2]['global_coord_system']['total'] 
            Sigma_tot = 1.0/3.0 * (sigma_tot_0+sigma_tot_1+sigma_tot_2)
            f1.write(str(i+1)+" "+str(Sigma_tot[0,0])+" "+str(Sigma_tot[0,1])+" "+str(Sigma_tot[0,2])\
            +" "+str(Sigma_tot[1,0])+" "+str(Sigma_tot[1,1])+" "+str(Sigma_tot[1,2])\
            +" "+str(Sigma_tot[2,0])+" "+str(Sigma_tot[2,1])+" "+str(Sigma_tot[2,2])+"\r\n")
        f1.write("$EndElementData\r\n")
        f1.close()
        
        return

    def get_K(self):
        return self.K
    def get_rhs(self):
        return self.rhs
    
    def post_processing_erreur(self,erreur,file_name):
       #copying mesh file
        with open(self.mesh_file) as f:
            with open(file_name+".msh", "w") as f1:
                for line in f:
                        f1.write(line)
        f.close()
        f1.close()
        f1 = open(file_name+".msh",'a')
        #vector displacement field at nodes
        f1.write("\r\n")
        f1.write("$NodeData\r\n")
        f1.write("1\n")
        f1.write('"Erreur displacement"\r\n')
        f1.write("1\r\n")
        f1.write("0.0\r\n")
        f1.write("3\r\n")
        f1.write("0\r\n")
        f1.write("3\r\n")
        f1.write(str(int(len(self.nodes)))+"\r\n")
        for i in range(len(self.nodes)):
            f1.write(str(i+1)+" "+str(erreur[6*i])+" "+str(erreur[6*i+1])+" "+str(erreur[6*i+2])+"\r\n")
        f1.write("$EndNodeData")    
        return
        
    def post_processing_Von_Mises(self, Sigma_VM, file_name):
       #copying mesh file
        with open(self.mesh_file) as f:
            with open(file_name+".msh", "w") as f1:
                for line in f:
                        f1.write(line)
        f.close()
        f1.close()
        f1 = open(file_name+".msh",'a')
        #vector displacement field at nodes
        f1.write("\r\n")
        f1.write("$NodeData\r\n")
        f1.write("1\r\n")
        f1.write('"Von Mises Stress"\r\n')
        f1.write("1\r\n")
        f1.write("0.0\r\n")
        f1.write("3\r\n")
        f1.write("0\r\n")
        f1.write("1\r\n")
        f1.write(str(int(len(self.nodes)))+"\r\n")
        for i in range(len(self.nodes)):
            f1.write(str(i+1)+" "+str(Sigma_VM[i])+"\r\n")
        f1.write("$EndNodeData")    
        return
    def post_processing_Von_Mises_reel(self, Sigma_VM, file_name):
       #copying mesh file
        with open(self.mesh_file) as f:
            with open(file_name+".msh", "w") as f1:
                for line in f:
                        f1.write(line)
        f.close()
        f1.close()
        f1 = open(file_name+".msh",'a')
        #vector displacement field at nodes
        f1.write("\r\n")
        f1.write("$NodeData\r\n")
        f1.write("1\n")
        f1.write('"Von Mises Stress (orig)"\r\n')
        f1.write("1\r\n")
        f1.write("0.0\r\n")
        f1.write("3\r\n")
        f1.write("0\r\n")
        f1.write("1\r\n")
        f1.write(str(int(len(self.nodes)))+"\r\n")
        for i in range(len(self.nodes)):
            f1.write(str(i+1)+" "+str(Sigma_VM[i])+"\r\n")
        f1.write("$EndNodeData")    
        return
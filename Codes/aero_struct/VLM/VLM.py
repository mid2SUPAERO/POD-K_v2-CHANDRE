"""
Python class that defines a VLM study.
The VLM uses vortex ring elements as described in 
J. Katz, A. Plotkin, Low-Speed Aerodynamics.

@ Sylvain DUBREUIL, ONERA
"""
import numpy as np
import scipy.linalg as sci_linalg
#This class allows to perform a vlm study in order to obtain the forces and 
#the CL and CD.
class VLM_study():
    #Inputs: \n
    #mesh_file: gmsh mesh file (preferably generates with wing_geo.py) \n
    #alpha: float, angle of attack in degree \n 
    #v_inf: float, freestream velocity in m.s^{-1} \n
    #rho: float, air density in kg.m^{-3} \n
    #symmetry : bool, True if the problem is symmetric
    def __init__(self,mesh_file, alpha = 2.5, le = [400,700], te = [200,500], v_inf = 0.84*295.4, rho = 0.3629 ,
                 x_wake= 1e10,symmetry = True):
        self.mesh_file = mesh_file
        self.alpha = alpha
        self.v_inf = v_inf
        self.rho = rho
        self.symmetry = symmetry
        self.x_wake = x_wake
        self.cosa = np.cos(self.alpha * np.pi / 180.)
        self.sina = np.sin(self.alpha * np.pi / 180.)
        self.tana = self.sina/self.cosa
        self.le = le
        self.te = te
        self.surfaces = self.read_gmsh_mesh()
        self.count_sing = 0
        self.beam_nodes = 0
        self.nodes = 0
    #reads the mesh file at gmsh format and create the list of 
    #lifting surfaces defined in the mesh. \n
    #Input: \n
    #self \n
    #Output: \n
    #surfaces: list, lifting surfaces \n
    #surfaces[i]: dict, keys \n
    #'n_panels': int, number of panels \n
    #'leading_edge': list, index of the panels that belong to the leading edge of the surface \n
    #'trailing_edge': list, index of the panels that belong to the trailing edge of the surface \n
    #'points': array [n_panels,4,3], coordinates of the points that define the panel \n
    #'ring_points': array [n_panels,4,3], coordinates of the points that define the vortex ring \n
    #'control_points': array [n_panels,3], coordinates of the control point where the tangency boundary condition is applied (3/4 of the chord) \n
    #'control_points_quart': array [n_panels,3], coordinates of the control point where the velocity/force is computed (1/4 of the chord) \n
    #'mesh', array [n_panels,7], definition of the mesh \n
    #'normals', array [n_panels,3], coordinates of the panel normal vector \n
    #'panel_pairs', array, pairs of panels that share a common bound segment \n
    #'areas', array[n_panels,1], wetted area of each panel \n              
    def read_gmsh_mesh(self):
        msh_input_file = self.mesh_file
        f=open(msh_input_file,'r')
        lines=f.readlines()
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
        self.nodes = nodes
        #Q4 elements and boundaries nodes
        ind_start_element = lines.index('$Elements\r\n')
        ind_stop_element = lines.index('$EndElements\r\n')
        n_elements = int(lines[ind_start_element+1])
        elements_tot = np.zeros((n_elements,7)) # elements_tot[:,0] = elements number, elements_tot[:,1] = elements type, elements_tot[:,2] = surfaces, elements_tot[:,3:] = nodes number 
        i=0
        for line in lines[ind_start_element+2:ind_stop_element]:
            split_line=line.split()
            elements_tot[i,0]=int(split_line[0])
            elements_tot[i,1]=int(split_line[1])
            elements_tot[i,2]=int(split_line[4])
            if elements_tot[i,1]==15:
                elements_tot[i,3]=int(split_line[5])
            elif elements_tot[i,1]==1:
                elements_tot[i,3]=int(split_line[5])
                elements_tot[i,4]=int(split_line[6])
            elif elements_tot[i,1]==2:
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
        #Q4 elements
        Q4_elements_ind = elements_tot[:,1] == 3
        Q4_elements = elements_tot[Q4_elements_ind,:]
        n_Q4 = Q4_elements.shape[0]
#        print "number of panels = ",n_Q4
        #Q4_elements[:,0] = np.arange(1,n_Q4+1)
        #Edges elements
        edges_elements_ind = elements_tot[:,1] == 1
        edges_elements = elements_tot[edges_elements_ind,:]
        n_edges_elements = edges_elements.shape[0]
        edges_elements[:,0] = np.arange(1,n_edges_elements+1)
        #Surfaces :
        n_surfaces = int(np.unique(Q4_elements[:,2]).shape[0])
        surfaces = []
        #Global vortex segments array
        self.global_segment_bounds = []
        self.global_control_points = []
        self.global_control_points_quart = []
        self.global_sum_index = []
        for i in range(n_surfaces):
            surfaces.append({'name':i+1,'mesh':Q4_elements[Q4_elements[:,2]==i+1,:],'n_panels':len(Q4_elements[Q4_elements[:,2]==i+1,:])})
            normals = np.zeros((surfaces[i]['n_panels'],3))
            control_points = np.zeros((surfaces[i]['n_panels'],3))
            control_points_quart = np.zeros((surfaces[i]['n_panels'],3))
            ring_points = np.zeros((surfaces[i]['n_panels'],4,3))
            points = np.zeros((surfaces[i]['n_panels'],4,3))
            areas = np.zeros((surfaces[i]['n_panels'],1))
            #leading and trailing edges 
            #TODO : reperer automatiquement la ligne qui correspond a la le et celle a la te
            nodes_le = edges_elements[edges_elements[:,2]==self.le[i],:]
            nodes_te = edges_elements[edges_elements[:,2]==self.te[i],:]                    
            leading_edge = []
            trailing_edge = []
            for j in range(surfaces[i]['n_panels']):
                panel = surfaces[i]['mesh'][j,:]
                panel_nodes = np.array([nodes[np.argwhere(nodes[:,0]==panel[3])[0,0],1:],\
                nodes[np.argwhere(nodes[:,0]==panel[4])[0,0],1:],\
                nodes[np.argwhere(nodes[:,0]==panel[5])[0,0],1:],\
                nodes[np.argwhere(nodes[:,0]==panel[6])[0,0],1:]])
                if np.array_equal(panel_nodes[:,1],np.ones((4,))*panel_nodes[0,1]) == False:#check if the 4 points are not on the same y plan (point that belongs to the joint)
                    partition = np.argpartition(panel_nodes[:,1],2)
                    A = panel_nodes[partition[:2]][np.argmin(panel_nodes[partition[:2],0]),:]
                    D = panel_nodes[partition[:2]][np.argmax(panel_nodes[partition[:2],0]),:]
                    B = panel_nodes[partition[2:]][np.argmin(panel_nodes[partition[2:],0]),:]
                    C = panel_nodes[partition[2:]][np.argmax(panel_nodes[partition[2:],0]),:]
                    normals[j,:] = np.cross(A-C,D-B)
                    norm = np.linalg.norm(normals[j,:])
                    normals[j,:] = normals[j,:]/norm
                else:
                    partition = np.argpartition(panel_nodes[:,2],2)
                    A = panel_nodes[partition[:2]][np.argmin(panel_nodes[partition[:2],0]),:]
                    D = panel_nodes[partition[:2]][np.argmax(panel_nodes[partition[:2],0]),:]
                    B = panel_nodes[partition[2:]][np.argmin(panel_nodes[partition[2:],0]),:]
                    C = panel_nodes[partition[2:]][np.argmax(panel_nodes[partition[2:],0]),:]
                    normals[j,:] = np.cross(A-C,B-D)
                    norm = np.linalg.norm(normals[j,:])
                    normals[j,:] = normals[j,:]/norm
                E = A+3.0/4.0*(D-A) 
                F = B+3.0/4.0*(C-B)
                control_points[j,:] = E+0.5*(F-E)
                ring_points[j,0,:] = A+1.0/4.0*(D-A)
                ring_points[j,1,:] = B+1.0/4.0*(C-B)
                ring_points[j,2,:] = B+5.0/4.0*(C-B)
                ring_points[j,3,:] = A+5.0/4.0*(D-A)
                control_points_quart[j,:] = ring_points[j,0,:]+0.5*(ring_points[j,1,:]-ring_points[j,0,:])
                areas[j,0] = 0.5*np.linalg.norm(np.cross(C-A,D-B)) #weighted areas
                points[j,0,:] = A
                points[j,1,:] = B
                points[j,2,:] = C
                points[j,3,:] = D
                #Check if the element belongs to a leading or trailing edge
                if np.in1d(nodes_le[:,3:5],panel[3:]).any() == True:
                    leading_edge.append(j)
                if np.in1d(nodes_te[:,3:5],panel[3:]).any() == True: 
                    trailing_edge.append(j)
                #Filling the global array
                #Segments points    
                self.global_segment_bounds.append([ring_points[j,0,:],ring_points[j,1,:]])
                self.global_segment_bounds.append([ring_points[j,1,:],ring_points[j,2,:]])
                self.global_segment_bounds.append([ring_points[j,2,:],ring_points[j,3,:]])
                self.global_segment_bounds.append([ring_points[j,3,:],ring_points[j,0,:]])
                if self.symmetry == False:
                    if np.in1d(nodes_te[:,3:5],panel[3:]).any() == True:
                        #Wake panel
                        C = ring_points[j,2,:].copy()
                        C[0] = self.x_wake
                        C[2] = C[0]*self.tana
                        D = ring_points[j,3,:].copy()
                        D[0] = self.x_wake
                        D[2] = D[0]*self.tana
                        self.global_segment_bounds.append([ring_points[j,3,:],ring_points[j,2,:]])
                        self.global_segment_bounds.append([ring_points[j,2,:],C])
                        self.global_segment_bounds.append([C,D])
                        self.global_segment_bounds.append([D,ring_points[j,3,:]])
                        #Summation index
                        self.global_sum_index.append(8)
                    else:
                        #Summation index
                        self.global_sum_index.append(4)                        
                else:
                    ring_A_sym = ring_points[j,0,:].copy()
                    ring_A_sym[1] = -ring_A_sym[1]
                    ring_B_sym = ring_points[j,1,:].copy()
                    ring_B_sym[1] = -ring_B_sym[1]
                    ring_C_sym = ring_points[j,2,:].copy()
                    ring_C_sym[1] = -ring_C_sym[1] 
                    ring_D_sym = ring_points[j,3,:].copy()
                    ring_D_sym[1] = -ring_D_sym[1]
                    self.global_segment_bounds.append([ring_B_sym,ring_A_sym])
                    self.global_segment_bounds.append([ring_A_sym,ring_D_sym])
                    self.global_segment_bounds.append([ring_D_sym,ring_C_sym])
                    self.global_segment_bounds.append([ring_C_sym,ring_B_sym])
                    if np.in1d(nodes_te[:,3:5],panel[3:]).any() == True:
                        #Wake panel
                        C = ring_points[j,2,:].copy()
                        C[0] = self.x_wake
                        C[2] = C[0]*self.tana
                        D = ring_points[j,3,:].copy()
                        D[0] = self.x_wake
                        D[2] = D[0]*self.tana
                        self.global_segment_bounds.append([ring_points[j,3,:],ring_points[j,2,:]])
                        self.global_segment_bounds.append([ring_points[j,2,:],C])
                        self.global_segment_bounds.append([C,D])
                        self.global_segment_bounds.append([D,ring_points[j,3,:]])
                        #Symmetry + Wake
                        ring_B_sym = ring_points[j,2,:].copy()
                        ring_A_sym = ring_points[j,3,:].copy()
                        ring_C_sym = C.copy()
                        ring_D_sym = D.copy()
                        ring_A_sym[1] = -ring_A_sym[1]
                        ring_B_sym[1] = -ring_B_sym[1]
                        ring_C_sym[1] = -ring_C_sym[1]
                        ring_D_sym[1] = -ring_D_sym[1]
                        self.global_segment_bounds.append([ring_B_sym,ring_A_sym])
                        self.global_segment_bounds.append([ring_A_sym,ring_D_sym])
                        self.global_segment_bounds.append([ring_D_sym,ring_C_sym])
                        self.global_segment_bounds.append([ring_C_sym,ring_B_sym])                        
                        #Summation index
                        self.global_sum_index.append(16)
                    else:
                        #Summation index
                        self.global_sum_index.append(8)
                #Control points        
                self.global_control_points.append(control_points[j,:])
                self.global_control_points_quart.append(control_points_quart[j,:])
                    
            surfaces[i]['normals'] = normals
            surfaces[i]['control_points'] = control_points
            surfaces[i]['control_points_quart'] = control_points_quart
            surfaces[i]['ring_points'] = ring_points
            surfaces[i]['areas'] = areas
            surfaces[i]['points'] = points
            surfaces[i]['leading_edge'] = leading_edge
            surfaces[i]['trailing_edge'] = trailing_edge 
            #Looking for equal segments (D_i,j & C_i,j == A_i+1,j & B_i+1,j)
            k = 0
            panel_candidate = np.arange(0,surfaces[i]['n_panels'],1)
            panel_pairs = []
            for panel in surfaces[i]['points']: 
                if (k in surfaces[i]['leading_edge']) == False:
                    A = panel[0,:]
                    B = panel[1,:] 
                    #looking for D_i
                    ind_1 = (surfaces[i]['points'][:,3,:] == A).all(axis=1)
                    #looking for C_i
                    ind_2 = (surfaces[i]['points'][:,2,:] == B).all(axis=1)
                    pair = panel_candidate[ind_1&ind_2]
                    panel_pairs.append([k,pair[0]])
                k=k+1    
            surfaces[i]['panel_pairs'] = np.array(panel_pairs)
            #Regularization : Panel pairs have to share common ring points and common quarter control points
            for pairs in surfaces[i]['panel_pairs']:
                surfaces[i]['ring_points'][pairs[1]][2] = surfaces[i]['ring_points'][pairs[0]][1]
                surfaces[i]['ring_points'][pairs[1]][3] = surfaces[i]['ring_points'][pairs[0]][0]
                
        self.global_control_points = np.array(self.global_control_points) 
        self.global_control_points_quart = np.array(self.global_control_points_quart)
        self.global_segment_bounds = np.array(self.global_segment_bounds)
        #self.global_sum_index.pop(-1)
        self.global_sum_index = np.array(self.global_sum_index)            
        return surfaces
    
    
    #calculates the induced speed for a vortex filament of intensity gamma = 1\n
    #Inputs: \n
    #self
    #A: array[3,:], coordinates of the filaments starting points \n
    #B: array[3,:], coordinates of the filaments ending points \n
    #P: array[3,:], coordinates of the points where the induced speed is calculated \n
    #Output: \n
    #v: array [3,:], induced speed at points P by the filaments A-B    
    def _calc_vorticity_vect(self,A,B,P):
        r1 = P - A
        r2 = P - B
        r0 = r1-r2
        r1_mag = np.linalg.norm(r1,axis = 1)
        r2_mag = np.linalg.norm(r2,axis = 1)     
        cross = np.cross(r1,r2)
        ind = r1_mag<1e-10 
        ind[r2_mag<1e-10] = True
        ind[np.linalg.norm(cross,axis=1)<1e-10] = True
        ind_true = np.invert(ind)
        dot_r1 = np.einsum('ij,ij->i', r0[ind_true], r1[ind_true])
        dot_r2 = np.einsum('ij,ij->i', r0[ind_true], r2[ind_true])
        a = (dot_r1/r1_mag[ind_true]-dot_r2/r2_mag[ind_true])/np.linalg.norm(cross[ind_true],axis=1)**2
        v = np.zeros((r1.shape[0],3))
        v[ind_true] = np.einsum('ij,i->ij', cross[ind_true], a)
        v = 1.0/(4.0*np.pi)*v
        return v

    
    #Assembling the AIC matrix and the induced speed tensor
    #Inputs: \n
    #self \n
    #c_pts, string, choice between '3/4' for tangency condition or '1/4' for computation of the induced speed at bound filament
    def _assemble_AIC_mtx(self, c_pts = '3/4'):        
        #We have to compute the induced speed of each segment at each control point
        #List of all the vortex segments
        segments = self.global_segment_bounds
        #List of the control points
        if c_pts == '3/4':
            points = self.global_control_points
        elif c_pts == '1/4':
            points = self.global_control_points_quart
        #Initialization of the global vector [segment begin, segment end, c_pts]
        vect_tot = np.zeros((points.shape[0]*segments.shape[0],3,3))
        #Fulfill the global vector with the segments coordinate
        vect_tot[:,0:2,:] = np.tile(segments,(points.shape[0],1,1))
        #Fulfill the global vector with the c_pts coordinate
        vect_tot[:,-1,:] = np.repeat(points,segments.shape[0],0)
        #Compute the induced speed vector
        induced_speed_vector_temp = self._calc_vorticity_vect(vect_tot[:,0],vect_tot[:,1],vect_tot[:,2])
        #print "induced_speed_vector_temp=",induced_speed_vector_temp
        #Sum the segment contribution according to the panel position
        sum_repeat = np.tile(self.global_sum_index,(points.shape[0]))
        sum_repeat = np.cumsum(sum_repeat)
        sum_repeat_new = np.zeros(sum_repeat.shape,dtype='int')
        sum_repeat_new[1:]=sum_repeat[0:-1]
        induced_speed_vector = np.add.reduceat(induced_speed_vector_temp, sum_repeat_new, axis=0)
        #Reshape to get the induced speed matrix
        induced_speed_matrix = induced_speed_vector.reshape((points.shape[0],points.shape[0],3))
        #Get the normal vectors
        normal_vectors = np.zeros((induced_speed_matrix.shape[0],3))
        i=0
        for surface in self.surfaces:
            normals = surface['normals']
            normal_vectors[i:i+normals.shape[0]]=normals
            i = i+normals.shape[0]
        
        AIC_matrix = np.einsum('ijk,ik->ij',induced_speed_matrix,normal_vectors)
        
        return AIC_matrix, induced_speed_matrix
    
    #creates the RHS \n
    #inputs: \n
    #self \n
    #outputs: \n
    #rhs: array [n_panels_tot], right hand side of the system allowing to find the circulation
    def _assemble_rhs(self):
        #vector of normal at each control points
        n_control_points = 0
        for surface in self.surfaces:
            n_control_points = n_control_points + len(surface['mesh'])    
        normals = np.zeros((n_control_points,3))
        i = 0
        for surface in self.surfaces:
            normals[i:i+len(surface['mesh'])] = surface['normals']
            i = i +  len(surface['mesh'])    
        #freestream vector
        cosa = np.cos(self.alpha * np.pi / 180.)
        sina = np.sin(self.alpha * np.pi / 180.)
        u = np.array([cosa, 0, sina])
        v = self.v_inf*u
        #computation of the rhs
        rhs = np.dot(normals,v)
        return rhs
        
    #solves the linear system in order to find the circulation of each ring vortex \n
    #input: \n
    #self \n
    #output: \n
    #gamme: array [n_panels_tot], vector of circulations    
    def compute_circulation(self):
        aic,_ = self._assemble_AIC_mtx()
        rhs = self._assemble_rhs() 
        aic_lu = sci_linalg.lu_factor(aic)
        gamma = sci_linalg.lu_solve(aic_lu,-rhs)
        return gamma          

    #computes local velocities and forces \n
    #inputs: \n
    #gamma: array [n_panels_tot], vector of circulations \n
    #outputs: \n
    #V: array [n_panels_tot,3] local velocity vector at each panel front bound segment \n
    #forces: [n_panels_tot,3] local force vector at each panel front bound segment \n
    def compute_velocities_and_forces(self,gamma):
        #first we compute an induced speed matrix with control points at the midpoint of each bound segment
        AIC_matrix,induced_speed_matrix =self. _assemble_AIC_mtx(c_pts='1/4') 
        #print "induced_speed_matrix=",induced_speed_matrix
        #Then we compute the speed due to the circulation gamma
        n_control_points = 0
        for surface in self.surfaces:
            n_control_points = n_control_points + len(surface['mesh'])
        V = np.zeros((n_control_points,3))
        for ind in range(3):
            V[:,ind] = np.dot(induced_speed_matrix[:,:,ind],gamma)  
        #We add the freestream velocity
        alpha = self.alpha * np.pi/180.0
        V[:,0] = V[:,0] + np.cos(alpha) * self.v_inf
        V[:,2] = V[:,2] + np.sin(alpha) * self.v_inf
        #once we get the velocity we can compute the local forces at the middle of each bound segment
        forces = np.zeros((n_control_points,3))
        j = 0
        ind_n_panels = 0
        for surface in self.surfaces:
            i = 0
            n_panels = len(surface['mesh'])
            for A,B,C,D in surface['ring_points']:
                bound = A-B
                cross = np.cross(bound,V[j,:])
                if (i in surface['leading_edge']) == True:
                    forces[j,:] = self.rho * gamma[j] * cross
                else:
                    ind = np.argwhere(surface['panel_pairs']==i)
                    ind_1 = surface['panel_pairs'][ind[0,0],:] 
                    ind_1 = ind_1+ind_n_panels
                    gamma_pairs = gamma[ind_1]
                    gamma_temp = gamma_pairs[0]-gamma_pairs[1]
                    forces[j,:] = self.rho*gamma_temp *cross
                i += 1    
                j += 1
            ind_n_panels+=n_panels    
        return V,forces   
        
    #computes total lift and drag \n
    #inputs: \n
    #forces: array [n_panels_tot,3] local force vector at each panel front bound segment \n 
    #outputs: \n
    #L: float, total lift N \n
    #D: float, total drag N \n
    def compute_L_D(self,forces):
        cosa = np.cos(self.alpha * np.pi / 180.)
        sina = np.sin(self.alpha * np.pi / 180.)
        # Compute the induced lift force on each lifting surface
        L = np.sum(-forces[:, 0] * sina + forces[:, 2] * cosa)
        # Compute the induced drag force on each lifting surface
        D = np.sum( forces[:, 0] * cosa + forces[:, 2] * sina)
        if self.symmetry:
            D*= 2.0
            L*= 2.0
        return L,D    
        
    #computes lift and drag coefficients \n
    #inputs: \n
    #L: float, total lift N \n
    #D: float, total drag N \n
    #outputs: \n
    #CL, float, lift coefficient \n
    #CD, float, drag coefficient \n               
    def compute_CL_CD(self,L,D):
        S_ref = 0
        for surface in self.surfaces:
            S_ref = S_ref+surface['areas'].sum()
        CL = L / (0.5 * self.rho * self.v_inf**2 * S_ref)
        CD = D / (0.5 * self.rho * self.v_inf**2 * S_ref)
        if self.symmetry == True:
            CL/=2.0
            CD/=2.0
        return CL,CD
    #post processing of the effort for visualization in GMSH
    #inputs: \n
    #gamma: array, circulations vector computed by compute_cirdulation\n    
    #forces: array, forces vector computed by compute_velocity_and_forces \n
    #output:\n
    #New mesh_file with post processing data
    def post_processing(self,file_name,gamma,forces):
        #copying mesh file
        with open(self.mesh_file) as f:
            with open('../results/'+file_name+".msh", "w") as f1:
                for line in f:
                        f1.write(line)
        f.close()
        f1.close()
        n_panels = 0
        for surface in self.surfaces:
            n_panels = n_panels + len(surface['mesh'])
        f=open('../results/'+file_name+".msh",'a')
        f.write('$ElementData\r\n')
        f.write('1\r\n')
        f.write('"Circulation"\r\n')
        f.write('1\r\n')
        f.write('0.0\r\n')
        f.write('3\r\n')
        f.write('0\r\n')
        f.write('1\r\n')
        f.write(str(int(n_panels))+'\r\n')
        j = 0
        for surface in self.surfaces:
            for i in range(len(surface['mesh'])):
                f.write(str(int(surface['mesh'][i,0]))+' %f \r\n' \
                %(gamma[j]))
                j = j + 1
        f.write('$EndElementData\r\n')
        f.write('$ElementData\r\n')
        f.write('1\r\n')
        f.write('"Force"\r\n')
        f.write('1\r\n')
        f.write('0.0\r\n')
        f.write('3\r\n')
        f.write('0\r\n')
        f.write('3\r\n')
        f.write(str(int(n_panels))+'\r\n')
        j = 0
        for surface in self.surfaces:
            for i in range(len(surface['mesh'])):
                f.write(str(int(surface['mesh'][i,0]))+' %f %f %f \r\n' \
                %(forces[j,0],forces[j,1],forces[j,2]))
                j = j + 1
        f.write('$EndElementData\r\n')
        f.write('$ElementData\r\n')
        f.write('1\r\n')
        f.write('"Delta P"\r\n')
        f.write('1\r\n')
        f.write('0.0\r\n')
        f.write('3\r\n')
        f.write('0\r\n')
        f.write('1\r\n')
        f.write(str(int(n_panels))+'\r\n')
        j = 0
        for surface in self.surfaces:
            for i in range(len(surface['mesh'])):
                f.write(str(int(surface['mesh'][i,0]))+' %f \r\n' \
                %(np.linalg.norm(forces[j,:])/surface['areas'][i]))
                j = j + 1
        f.write('$EndElementData\r\n')
        f.close()        
        return
        
    def compute_equivalent_beam_nodes(self,fem_origin):
        #forces are compute at the 1/4 of each panel
        beam_nodes = []
        # loop over the surfaces 
        n_panels = 0
        for surface in self.surfaces:
            #mesh columns
            n_c = len(surface['leading_edge'])
            #mesh lines
            n_l = surface['control_points_quart'].shape[0]/n_c
            #loop over the column
            for i in range(n_c):
                #find all panels of the column
                y_l_e_i = surface['control_points_quart'][surface['leading_edge'][i],1]
                j = 0
                ind_c_i = []
                for point in surface['control_points_quart']:
                    if np.allclose(y_l_e_i,point[1],1e-9,1e-9):
                        ind_c_i.append(n_panels+j)
                    j = j + 1
                #check that all panels have been find
                if len(ind_c_i) != n_l:
                    print "WARNING: MISSING PANELS IN "+str(surface['name'])+" COLUMN "+str(i)
                #beam node
                #Panel at the leading edge for this column
                panel_le_c_i = surface['points'][surface['leading_edge'][i]]
                #Middle point of leading edge
                x0 = panel_le_c_i[0]+(panel_le_c_i[1]-panel_le_c_i[0])/2.0 
                #Panel at the trailing edge for this column
                panel_te_c_i = surface['points'][surface['trailing_edge'][i]]
                #Middle point of trailing edge
                x1 = panel_te_c_i[3]+(panel_le_c_i[2]-panel_le_c_i[3])/2.0
                #beam node is located at fem origin of the vector x0x1
                beam_node = x0+fem_origin*(x1-x0)
                beam_nodes.append(beam_node)                                
            n_panels = n_panels+(n_c*n_l)
        return np.array(beam_nodes)
             
    def compute_force_and_moment_equivalent_beam(self,beam_nodes,forces):        
        #forces are compute at the 1/4 of each panel
        nodal_forces = []
        nodal_moments = []
        # loop over the surfaces 
        n_panels = 0
        k = 0
        for surface in self.surfaces:
            #mesh columns
            n_c = len(surface['leading_edge'])
            #mesh lines
            n_l = surface['control_points_quart'].shape[0]/n_c
            #loop over the column
            for i in range(n_c):
                #find all panels of the column
                y_l_e_i = surface['control_points_quart'][surface['leading_edge'][i],1]
                j = 0
                ind_c_i = []
                for point in surface['control_points_quart']:
                    if np.allclose(y_l_e_i,point[1],1e-9,1e-9):
                        ind_c_i.append(n_panels+j)
                    j = j + 1
                #check that all panels have been find
                if len(ind_c_i) != n_l:
                    print "WARNING: MISSING PANELS IN "+str(surface['name'])+" COLUMN "+str(i)
                #beam node
                beam_node = beam_nodes[k]
                #Force and moment resultant
                force_c_i = np.sum(forces[ind_c_i,:],axis = 0)
                d_c_i = surface['control_points_quart'][np.array(ind_c_i)-n_panels]-beam_node
                moment_c_i = np.cross(d_c_i,forces[ind_c_i,:]).sum(axis=0)
                nodal_forces.append(force_c_i)
                nodal_moments.append(moment_c_i) 
                k=k+1               
            n_panels = n_panels+(n_c*n_l)             
        return np.array(nodal_forces), np.array(nodal_moments)
        
    def resultant_force_and_moment(self,x,forces):
        force = np.sum(forces,axis=0)
        d = self.global_control_points_quart-x
        moment = np.cross(d,forces).sum(axis=0)
        return force, moment
    
    def compute_force_at_nodes(self,nodes,forces):
        #forces are computed at the middle 1/4 chord of each panel, we have to project this forces at nodes
        forces_at_nodes = np.zeros((nodes.shape[0],3))
        i = 0
        for surface in self.surfaces:
            for points in surface['mesh']:
                A = points[3]-1
                B = points[4]-1
                C = points[5]-1
                D = points[6]-1
                for x in [A,B,C,D]:
                    x = int(x)
                    forces_at_nodes[x,:] = forces_at_nodes[x,:]+forces[i,:]/4.0
                i = i + 1    
        return forces_at_nodes            
    
    def deformed_mesh_file(self,new_nodes,mesh_file_out):
        #copying the mesh file and replacing the points coordinates by the new one
        f=open(self.mesh_file,'r')
        lines=f.readlines()
        f.close()
        #nodes
        ind_start_node = lines.index('$Nodes\r\n')
        ind_stop_node = lines.index('$EndNodes\r\n')
        f = open(mesh_file_out,'w')
        for i in range(ind_start_node+1):
            f.write(lines[i])  
        f.write(str(len(new_nodes))+'\r\n')    
        for i in range(len(new_nodes)):
            f.write(str(int(new_nodes[i,0]))+' '+str(new_nodes[i,1])+' '+str(new_nodes[i,2])+' '\
            +str(new_nodes[i,3])+'\r\n')
        for line in lines[ind_stop_node:]:
            f.write(line)
        f.close()
        return 
    
    def get_B(self):
        rhs = self._assemble_rhs() 
        return rhs
    
    def get_A(self):
        AIC_matrix,induced_speed_matrix =self._assemble_AIC_mtx(c_pts='1/4')
        return AIC_matrix

    def compute_forces_ROB(self,gamma):
        #first we compute an induced speed matrix with control points at the midpoint of each bound segment
        _,induced_speed_matrix =self. _assemble_AIC_mtx(c_pts='1/4') 
        #print "induced_speed_matrix=",induced_speed_matrix
        #Then we compute the speed due to the circulation gamma
        n_control_points = 0
        for surface in self.surfaces:
            n_control_points = n_control_points + len(surface['mesh'])
        V = np.zeros((n_control_points,3))
        for ind in range(3):
            V[:,ind] = np.dot(induced_speed_matrix[:,:,ind],gamma)
        #We add the freestream velocity
        alpha = self.alpha * np.pi/180.0
        V[:,0] = V[:,0] + np.cos(alpha) * self.v_inf
        V[:,2] = V[:,2] + np.sin(alpha) * self.v_inf
        #once we get the velocity we can compute the local forces at the middle of each bound segment
        forces = np.zeros((n_control_points,3))
        j = 0
        ind_n_panels = 0
        for surface in self.surfaces:
            i = 0
            n_panels = len(surface['mesh'])
            for A,B,C,D in surface['ring_points']:
                bound = A-B
                cross = np.cross(bound,V[j,:])
                if (i in surface['leading_edge']) == True:
                    forces[j,:] = self.rho * gamma[j] * cross
                else:
                    ind = np.argwhere(surface['panel_pairs']==i)
                    ind_1 = surface['panel_pairs'][ind[0,0],:] 
                    ind_1 = ind_1+ind_n_panels
                    gamma_pairs = gamma[ind_1]
                    gamma_temp = gamma_pairs[0]-gamma_pairs[1]
                    forces[j,:] = self.rho*gamma_temp *cross
                i += 1    
                j += 1
            ind_n_panels+=n_panels    
        return forces   
            
    def post_processing_gamma(self,file_name,gamma):
        #copying mesh file
        with open(self.mesh_file) as f:
            with open('../results/'+file_name+".msh", "w") as f1:
                for line in f:
                        f1.write(line)
        f.close()
        f1.close()
        n_panels = 0
        for surface in self.surfaces:
            n_panels = n_panels + len(surface['mesh'])
        f=open('../results/'+file_name+".msh",'a')
        f.write('$ElementData\r\n')
        f.write('1\r\n')
        f.write('"Circulation (mean)"\r\n')
        f.write('1\r\n')
        f.write('0.0\r\n')
        f.write('3\r\n')
        f.write('0\r\n')
        f.write('1\r\n')
        f.write(str(int(n_panels))+'\r\n')
        j = 0
        for surface in self.surfaces:
            for i in range(len(surface['mesh'])):
                f.write(str(int(surface['mesh'][i,0]))+' %f \r\n' %(gamma[j]))
                j = j + 1
        f.write('$EndElementData\r\n')
        f.close()
        return
    def post_processing_gamma_reel(self,file_name,gamma):
        #copying mesh file
        with open(self.mesh_file) as f:
            with open('../results/'+file_name+".msh", "w") as f1:
                for line in f:
                        f1.write(line)
        f.close()
        f1.close()
        n_panels = 0
        for surface in self.surfaces:
            n_panels = n_panels + len(surface['mesh'])
        f=open('../results/'+file_name+".msh",'a')
        f.write('$ElementData\r\n')
        f.write('1\n')
        f.write('"Circulation (orig)"\r\n')
        f.write('1\r\n')
        f.write('0.0\r\n')
        f.write('3\r\n')
        f.write('0\r\n')
        f.write('1\r\n')
        f.write(str(int(n_panels))+'\r\n')
        j = 0
        for surface in self.surfaces:
            for i in range(len(surface['mesh'])):
                f.write(str(int(surface['mesh'][i,0]))+' %f \r\n' %(gamma[j]))
                j = j + 1
        f.write('$EndElementData\r\n')
        f.close()
        return
    def post_processing_gamma_var1(self,file_name,gamma):
        #copying mesh file
        with open(self.mesh_file) as f:
            with open('../results/'+file_name+".msh", "w") as f1:
                for line in f:
                        f1.write(line)
        f.close()
        f1.close()
        n_panels = 0
        for surface in self.surfaces:
            n_panels = n_panels + len(surface['mesh'])
        f=open('../results/'+file_name+".msh",'a')
        f.write('$ElementData\r\n')
        f.write('1\r\n')
        f.write('"Circulation (mean+3*var^2)"\r\n')
        f.write('1\r\n')
        f.write('0.0\r\n')
        f.write('3\r\n')
        f.write('0\r\n')
        f.write('1\r\n')
        f.write(str(int(n_panels))+'\r\n')
        j = 0
        for surface in self.surfaces:
            for i in range(len(surface['mesh'])):
                f.write(str(int(surface['mesh'][i,0]))+' %f \r\n' %(gamma[j]))
                j = j + 1
        f.write('$EndElementData\r\n')
        f.close()
        return
    def post_processing_gamma_var2(self,file_name,gamma):
        #copying mesh file
        with open(self.mesh_file) as f:
            with open('../results/'+file_name+".msh", "w") as f1:
                for line in f:
                        f1.write(line)
        f.close()
        f1.close()
        n_panels = 0
        for surface in self.surfaces:
            n_panels = n_panels + len(surface['mesh'])
        f=open('../results/'+file_name+".msh",'a')
        f.write('$ElementData\r\n')
        f.write('1\r\n')
        f.write('"Circulation (mean-3*var^2)"\r\n')
        f.write('1\r\n')
        f.write('0.0\r\n')
        f.write('3\r\n')
        f.write('0\r\n')
        f.write('1\r\n')
        f.write(str(int(n_panels))+'\r\n')
        j = 0
        for surface in self.surfaces:
            for i in range(len(surface['mesh'])):
                f.write(str(int(surface['mesh'][i,0]))+' %f \r\n' %(gamma[j]))
                j = j + 1
        f.write('$EndElementData\r\n')
        f.close()
        return
    
    def post_processing_erreur(self,file_name,erreur):
        #copying mesh file
        with open(self.mesh_file) as f:
            with open('../results/'+file_name+".msh", "w") as f1:
                for line in f:
                        f1.write(line)
        f.close()
        f1.close()
        n_panels = 0
        for surface in self.surfaces:
            n_panels = n_panels + len(surface['mesh'])
        f=open('../results/'+file_name+".msh",'a')
        f.write('$ElementData\r\n')
        f.write('1\r\n')
        f.write('"Erreur Circulation"\r\n')
        f.write('1\r\n')
        f.write('0.0\r\n')
        f.write('3\r\n')
        f.write('0\r\n')
        f.write('1\r\n')
        f.write(str(int(n_panels))+'\r\n')
        j = 0
        for surface in self.surfaces:
            for i in range(len(surface['mesh'])):
                f.write(str(int(surface['mesh'][i,0]))+' %f \r\n' %(erreur[j]))
                j = j + 1
        f.write('$EndElementData\r\n')
        f.close()
        return                
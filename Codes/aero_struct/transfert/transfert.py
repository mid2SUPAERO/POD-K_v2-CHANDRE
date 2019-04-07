import numpy as np
from rbf_poly import Rbf_poly
from scipy.interpolate import Rbf
"""
Definition of the RBF interpolation matrices used to transfer
structural displacement to aero displacement u_a = H*u_s
aero forces to structural forces F_s = H^t*F_a

@ Sylvain DUBREUIL, ONERA
"""
def transfert_matrix(nodes_s,nodes_a, function_type = 'gaussian',epsilon = 1.0):
    #Create an RBF interpolation with polynomial terms from the structural nodes and aerodynamic points coordinates
    inter = Rbf_poly(nodes_s[:, 0], nodes_s[:, 1], nodes_s[:, 2],
                     nodes_a[:, 0], nodes_a[:, 1], nodes_a[:, 2], function=function_type, epsilon=epsilon)
    return inter.H

"""
Definition of the deformed VLM mesh for the beam model
"""
def beam_transfert(mesh_file,beam_nodes,u_a,fem_origin):
    #read gmsh msh
    f=open(mesh_file,'r')
    lines=f.readlines()
    f.close()
    #nodes
    ind_start_node = lines.index('$Nodes\n')
    ind_stop_node = lines.index('$EndNodes\n')
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
    #u_a are at the middle of panels column, values at column border are obtained thanks to linear interpolation

    u_z = Rbf(beam_nodes[:,0],beam_nodes[:,1],beam_nodes[:,2],u_a[2:-1:6])
    t_y = Rbf(beam_nodes[:,0],beam_nodes[:,1],beam_nodes[:,2],u_a[4:-1:6])
    #find nodes that belongs to each sections
    sections = []    
    ind_sort = np.argsort(nodes[:,2])
    nodes = nodes[ind_sort]    
    ind = 0   
    while ind<len(nodes):
        ind_bool = np.isclose(nodes[ind,2],nodes[:,2])
        ind_where = np.argwhere(ind_bool == True)
        sections.append(nodes[ind_where])
        ind = ind_where[-1]+1
        
    #apply rotation and translation   
    for section in sections[1:]:
        #x_max and x_min section
        ind = section[:,0,1].argmin()
        x_min_point = section[ind,0]
        ind = section[:,0,1].argmax()
        x_max_point = section[ind,0]
        beam_point = x_min_point[1:]+(x_max_point[1:]-x_min_point[1:])*fem_origin
        #interpolation of theta and z at beam point
        theta_i = t_y(beam_point[0],beam_point[1],beam_point[2])
        z_i = u_z(beam_point[0],beam_point[1],beam_point[2])
        #apply theta_i to the section (rotation about the beam_point)        
        d = section[:,0,1:]-beam_point           
        #rotation
        theta_tot = theta_i    
        #new x z coord
        x_new = beam_point[0]+d[:,0]*np.cos(theta_i)-d[:,2]*np.sin(theta_tot)
        z_new = beam_point[2]+d[:,0]*np.sin(theta_i)+d[:,2]*np.cos(theta_tot) 
        section[:,0,1] = x_new.copy()
        section[:,0,3] = z_new.copy()+z_i
    nodes_new = np.zeros(nodes.shape)
    i = 0
    for section in sections:
        for node in section[:,0,:]:
            nodes_new[i,:] = node
            i =i+1
    nodes = nodes_new.copy()
    return nodes    

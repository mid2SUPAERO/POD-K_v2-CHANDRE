import numpy as np
from define_geo import airbus_wing

#definition of .geo file for creation of a vlm wing mesh
def create_VLM_mesh(span_1,span_2,theta_1,theta_2,theta_3,L_1,c_1,c_2,c_3,phi_1,phi_2,d_1,d_2,n_1 = 12, n_2 = 20, n_3 = 39):
    #plan form is defined by 2 surfaces i.e 6 points
    points = np.zeros((6,3))
    l = c_1 - 0.25*L_1
    points[0,:] = [-L_1*0.25,0.0,np.tan(theta_1)*0.25*L_1]
    points[1,:] = [l,0.0,-np.tan(theta_1)*l]
    delta_z_1 = np.tan(d_1)*span_1
    points[2,:] = [np.tan(phi_1)*span_1 +0.75*c_2 ,span_1,delta_z_1-0.75*c_2*np.tan(theta_2)]
    points[3,:] = [np.tan(phi_1)*span_1-0.25*c_2 ,span_1,delta_z_1+0.25*c_2*np.tan(theta_2)]
    delta_z_2 = delta_z_1 + np.tan(d_2)*span_2
    delta_x = np.tan(phi_1)*span_1
    points[4,:] = [delta_x+np.tan(phi_2)*span_2+0.75*c_3,span_1+span_2,delta_z_2-0.75*c_3*np.tan(theta_3)]
    points[5,:] = [delta_x+np.tan(phi_2)*span_2-0.25*c_3,span_1+span_2,delta_z_2+0.25*c_2*np.tan(theta_3)]
    #mesh nodes
    #Writing the .geo file for the vlm mesh
    f = open(r'../mesh/param_wing/VLM_mesh.geo','w')
    for i in range(6):
            f.write('Point(100'+str(int(i+1))+') = {'+str(points[i,0])+','+str(points[i,1])+','+str(points[i,2])+',1e+5};\n')
    
    f.write('Line(100) = {1001,1002};\n')
    f.write('Line(200) = {1002,1003};\n')
    f.write('Line(300) = {1003,1004};\n')
    f.write('Line(400) = {1004,1001};\n')
    f.write('Line(500) = {1003,1005};\n')
    f.write('Line(600) = {1005,1006};\n')
    f.write('Line(700) = {1006,1004};\n')
    f.write('Physical Line("leading_edge") = {400,700};\n')
    f.write('Physical Line("trailing_edge") = {200,500};\n')
    f.write('Line Loop(100) = {100,200,300,400};\n')
    f.write('Line Loop(200) = {-300,500,600,700};\n')
    f.write('Ruled Surface(1) = {100};\n')
    f.write('Ruled Surface(2) = {200};\n')
    f.write('Physical Surface("wing") = {1, 2};\n')
    f.write('Transfinite Line {100,300,600} = '+str(n_1)+' Using Progression 1;\n')
    f.write('Transfinite Line {200,400} = '+str(n_2)+' Using Progression 1;\n')
    f.write('Transfinite Line {500,700} = '+str(n_3)+' Using Progression 1;\n')
    f.write('Transfinite Surface {1} = {1001, 1002, 1003, 1004};\n')
    f.write('Transfinite Surface {2} = {1004, 1003, 1005, 1006};\n')
    f.write('Recombine Surface {1,2};\n')
    f.close()
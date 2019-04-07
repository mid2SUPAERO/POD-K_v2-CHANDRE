import numpy as np
from define_geo import airbus_wing

#definition of .geo file for fast of a fem wing box

def create_FEM_mesh(VLM_geo,tck,chord_pos, n_ribs_1,n_ribs_2,n_x = 8,n_y = 2,n_z = 3):
    f = open(VLM_geo,'r')
    lines = f.readlines()
    f.close()
    points_geo = np.zeros((6,3))
    for i in range(6):
        line = lines[i].split()
        coord = line[2].split(',')
        points_geo[i,0] = float(coord[0].split('{')[1])
        points_geo[i,1] = float(coord[1])
        points_geo[i,2] = float(coord[2])
        
    #wing box is defined by 3 boxes located at the sections of the wing
    points = np.zeros((12,3))
    for i in range(3):
        points_A_B = points_geo[(2*i):(2*i)+2,:]
        ind = points_A_B[:,0].argsort()
        points_A_B = points_A_B[ind,:]
        vect = points_A_B[1]-points_A_B[0]
        A_s = points_A_B[0] + chord_pos[i,0]*vect
        B_s = points_A_B[0] + chord_pos[i,1]*vect
        points[(4*i):(4*i)+2,:] = A_s
        points[(4*i)+2:(4*i)+4,:] = B_s
        points[(4*i),2] = points[(4*i),2]+tck[i,0]/2.0 
        points[(4*i)+1,2] = points[(4*i)+1,2]-tck[i,0]/2.0
        points[(4*i)+2,2] = points[(4*i)+2,2]-tck[i,1]/2.0
        points[(4*i)+3,2] = points[(4*i)+3,2]+tck[i,1]/2.0
    
    #this boxes defined the front and rear spars and the upper and lower skin
    #definition of the ribs
    #vector 1-5
    v15 = points[4,:]-points[0,:]
    delta_15 = v15/n_ribs_1
    points_15 = np.zeros((n_ribs_1-1,3))
    for i in range(n_ribs_1-1):
        points_15[i,:] = points[0,:]+(i+1)*delta_15
    #vector 4-8
    v48 = points[7,:]-points[3,:]
    delta_48 = v48/n_ribs_1
    points_48 = np.zeros((n_ribs_1-1,3))
    for i in range(n_ribs_1-1):
        points_48[i,:] = points[3,:]+(i+1)*delta_48
    #vector 2-6
    v26 = points[5,:]-points[1,:]
    delta_26 = v26/n_ribs_1
    points_26 = np.zeros((n_ribs_1-1,3))
    for i in range(n_ribs_1-1):
        points_26[i,:] = points[1,:]+(i+1)*delta_26
    #vector 3-7
    v37 = points[6,:]-points[2,:]
    delta_37 = v37/n_ribs_1
    points_37 = np.zeros((n_ribs_1-1,3))
    for i in range(n_ribs_1-1):
        points_37[i,:] = points[2,:]+(i+1)*delta_37
        
    #vector 5-9
    v59 = points[8,:]-points[4,:]
    delta_59 = v59/n_ribs_2
    points_59 = np.zeros((n_ribs_2-1,3))
    for i in range(n_ribs_2-1):
        points_59[i,:] = points[4,:]+(i+1)*delta_59
    
    #vector 6-10
    v610 = points[9,:]-points[5,:]
    delta_610 = v610/n_ribs_2
    points_610 = np.zeros((n_ribs_2-1,3))
    for i in range(n_ribs_2-1):
        points_610[i,:] = points[5,:]+(i+1)*delta_610
    
    #vector 7-11
    v711 = points[10,:]-points[6,:]
    delta_711 = v711/n_ribs_2
    points_711 = np.zeros((n_ribs_2-1,3))
    for i in range(n_ribs_2-1):
        points_711[i,:] = points[6,:]+(i+1)*delta_711
    
    #vector 8-12
    v812 = points[11,:]-points[7,:]
    delta_812 = v812/n_ribs_2
    points_812 = np.zeros((n_ribs_2-1,3))
    for i in range(n_ribs_2-1):
        points_812[i,:] = points[7,:]+(i+1)*delta_812    
    
    #
    points_tot = np.zeros((12+4*(n_ribs_1-1)+4*(n_ribs_2-1),3))
    vects_1 = [points_15,points_48,points_26,points_37]
    points_tot[0:12,:] = points
    for i in range(4):
        points_tot[12+i*(n_ribs_1-1):12+(i+1)*(n_ribs_1-1),:] = vects_1[i] 
    vects_2 = [points_59,points_610,points_711,points_812]
    points_tot[0:12,:] = points
    for i in range(4):
        points_tot[12+4*(n_ribs_1-1)+i*(n_ribs_2-1):12+4*(n_ribs_1-1)+(i+1)*(n_ribs_2-1),:] = vects_2[i]
    
    ind = points_tot[:,1].argsort()
    points_tot = points_tot[ind]
    points_final = np.zeros(points_tot.shape)
    for i in range(points_tot.shape[0]/4):
        quad_points = points_tot[i*4:(i+1)*4,:]
        ind_z = quad_points[:,2].argsort()
        points_final[i*4,:] = quad_points[ind_z][2:,:][quad_points[ind_z][2:,0].argmin()]
        points_final[i*4+1,:] = quad_points[ind_z][:2,:][quad_points[ind_z][:2,0].argmin()]
        points_final[i*4+2,:] = quad_points[ind_z][:2,:][quad_points[ind_z][:2,0].argmax()]
        points_final[i*4+3,:] = quad_points[ind_z][2:,:][quad_points[ind_z][2:,0].argmax()]
    
        
    #number of mesh nodes
    #Writing the .geo file for the vlm mesh
    f = open(r'../mesh/param_wing/FEM_mesh.geo','w')
    for i in range(points_final.shape[0]):
            f.write('Point('+str(int(i+1))+') = {'+str(points_final[i,0])+','+str(points_final[i,1])+','+str(points_final[i,2])+',1e+1};\n')
    #lines
    for i in range(points_final.shape[0]/4-1):
        f.write('Line('+str(8*i+1)+') = {'+str(4*i+1)+','+str(4*i+2)+'};\n')
        f.write('Line('+str(8*i+2)+') = {'+str(4*i+2)+','+str(4*i+3)+'};\n')
        f.write('Line('+str(8*i+3)+') = {'+str(4*i+3)+','+str(4*i+4)+'};\n')
        f.write('Line('+str(8*i+4)+') = {'+str(4*i+4)+','+str(4*i+1)+'};\n')
        
        f.write('Line('+str(8*i+5)+') = {'+str(4*i+1)+','+str(4*(i+1)+1)+'};\n')
        f.write('Line('+str(8*i+6)+') = {'+str(4*i+2)+','+str(4*(i+1)+2)+'};\n')
        f.write('Line('+str(8*i+7)+') = {'+str(4*i+3)+','+str(4*(i+1)+3)+'};\n')
        f.write('Line('+str(8*i+8)+') = {'+str(4*i+4)+','+str(4*(i+1)+4)+'};\n')
    
    #4 last lines
    ind = points_final.shape[0]/4-1
    f.write('Line('+str(8*ind+1)+') = {'+str(4*ind+1)+','+str(4*ind+2)+'};\n')
    f.write('Line('+str(8*ind+2)+') = {'+str(4*ind+2)+','+str(4*ind+3)+'};\n')
    f.write('Line('+str(8*ind+3)+') = {'+str(4*ind+3)+','+str(4*ind+4)+'};\n')
    f.write('Line('+str(8*ind+4)+') = {'+str(4*ind+4)+','+str(4*ind+1)+'};\n')
    
    #Surfaces
    skins = []
    spars_le = []
    spars_te = []
    ribs = []
    #5 surfaces per boxe
    for i in range(points_final.shape[0]/4-1):
        #skins
        f.write('Line Loop('+str(5*i+1)+') = {'+str(8*i+4)+','+str(8*i+5)+',-'+str(8*i+12)+',-'+str(8*i+8)+'};\n')
        f.write('Surface('+str(5*i+1)+') = {'+str(5*i+1)+'};\n')
        f.write('Line Loop('+str(5*i+2)+') = {'+str(8*i+2)+','+str(8*i+7)+',-'+str(8*i+10)+',-'+str(8*i+6)+'};\n')
        f.write('Surface('+str(5*i+2)+') = {'+str(5*i+2)+'};\n')  
        skins.extend([5*i+1,5*i+2])
        
        #spars
        f.write('Line Loop('+str(5*i+3)+') = {'+str(8*i+3)+','+str(8*i+8)+',-'+str(8*i+11)+',-'+str(8*i+7)+'};\n')
        f.write('Surface('+str(5*i+3)+') = {'+str(5*i+3)+'};\n')
        f.write('Line Loop('+str(5*i+4)+') = {'+str(8*i+1)+','+str(8*i+6)+',-'+str(8*i+9)+',-'+str(8*i+5)+'};\n')
        f.write('Surface('+str(5*i+4)+') = {'+str(5*i+4)+'};\n')    
        spars_te.extend([5*i+3])
        spars_le.extend([5*i+4])
        
        #ribs
        f.write('Line Loop('+str(5*i+5)+') = {'+str(8*i+10)+','+str(8*i+11)+','+str(8*i+12)+','+str(8*i+9)+'};\n')
        f.write('Surface('+str(5*i+5)+') = {'+str(5*i+5)+'};\n')    
        ribs.extend([5*i+5])
    
    f.write('Physical Surface("skins") = {')
    for i in range(len(skins)-1):
        f.write(str(skins[i])+',')
    f.write(str(skins[-1])+'};\n')
    
    f.write('Physical Surface("spars_le") = {')
    for i in range(len(spars_le)-1):
        f.write(str(spars_le[i])+',')
    f.write(str(spars_le[-1])+'};\n')  

    f.write('Physical Surface("spars_te") = {')
    for i in range(len(spars_te)-1):
        f.write(str(spars_te[i])+',')
    f.write(str(spars_te[-1])+'};\n')  
    
    f.write('Physical Surface("ribs") = {')
    for i in range(len(ribs)-1):
        f.write(str(ribs[i])+',')
    f.write(str(ribs[-1])+'};\n')  
    
    #mesh
    f.write('Transfinite Line {')
    for i in range(points_final.shape[0]/4-1):
        f.write(str(8*i+4)+','+str(8*i+2)+',')
    ind = points_final.shape[0]/4-1
    f.write(str(8*ind+4)+','+str(8*ind+2)+'} = '+str(n_x)+' Using Progression 1;\n')
    
    f.write('Transfinite Line {')
    for i in range(points_final.shape[0]/4-1):
        f.write(str(8*i+3)+','+str(8*i+1)+',')
    ind = points_final.shape[0]/4-1
    f.write(str(8*ind+3)+','+str(8*ind+1)+'} = '+str(n_z)+' Using Progression 1;\n')
    
    f.write('Transfinite Line {')
    for i in range(points_final.shape[0]/4-1):
        f.write(str(8*i+7)+','+str(8*i+8)+',')
    ind = points_final.shape[0]/4-1
    f.write(str(8*ind+7)+','+str(8*ind+8)+'} = '+str(n_y)+' Using Progression 1;\n')
    
    f.write('Transfinite Line {')
    for i in range(points_final.shape[0]/4-1):
        f.write(str(8*i+5)+','+str(8*i+6)+',')
    ind = points_final.shape[0]/4-1
    f.write(str(8*ind+5)+','+str(8*ind+6)+'} = '+str(n_y)+'Using Progression 1;\n')
    
    #structured mesh transfinite surface
    #5 surfaces per boxe
    for i in range(points_final.shape[0]/4-1):
        #skins
        f.write('Transfinite Surface('+str(5*i+1)+') = {'+str(4*i+1)+','+str(4*i+4)+',-'+str(4*i+8)+',-'+str(4*i+5)+'};\n')
        f.write('Transfinite Surface('+str(5*i+2)+') = {'+str(4*i+2)+','+str(4*i+3)+',-'+str(4*i+7)+',-'+str(4*i+6)+'};\n')
        #spars
        f.write('Transfinite Surface('+str(5*i+3)+') = {'+str(4*i+4)+','+str(4*i+3)+',-'+str(4*i+7)+',-'+str(4*i+8)+'};\n')
        f.write('Transfinite Surface('+str(5*i+4)+') = {'+str(4*i+1)+','+str(4*i+2)+',-'+str(4*i+6)+',-'+str(4*i+5)+'};\n')
        #rib
        f.write('Transfinite Surface('+str(5*i+5)+') = {'+str(4*i+5)+','+str(4*i+6)+',-'+str(4*i+7)+',-'+str(4*i+8)+'};\n')   
    
    f.close()
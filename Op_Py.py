"""
Disclaimer:

This code is designed with several simplifications to enhance its usability and performance. While it has been tested and utilized in published research, it may not cover all potential scenarios and edge cases. Users should evaluate and validate the results based on their specific needs and requirements. The code is provided to support further research and development, and contributions for improvements are encouraged.

"""

# OpenSees Functions
import numpy as np
import openseespy.opensees as ops
import opsvis as opsv
import matplotlib.pyplot as plt

def prepare_ops(indata, xrang, yrang, zrang, mat_prop, notrot, www, Curve):
  E = mat_prop[0]
  G = mat_prop[1]
  indata_cop=indata.copy()

  avvrXw=np.mean((indata_cop[(indata_cop[:,15]==100) | (indata_cop[:,15]==404)])[:,0])
  avvrYw=np.mean((indata_cop[indata_cop[:,15]==10])[:,0])
  avvrZw=np.mean((indata_cop[indata_cop[:,15]==1])[:,0])
  avvrDw=np.mean((indata_cop[indata_cop[:,20]==405])[:,0])

  avvrXl=np.mean((indata_cop[(indata_cop[:,15]==100) | (indata_cop[:,15]==404)])[:,1])
  avvrYl=np.mean((indata_cop[indata_cop[:,15]==10])[:,1])
  avvrZl=np.mean((indata_cop[indata_cop[:,15]==1])[:,1])
  avvrDl=np.mean((indata_cop[indata_cop[:,20]==405])[:,1])

  avvrXh=abs(np.mean((indata_cop[(indata_cop[:,15]==100) | (indata_cop[:,15]==404)])[:,19]))
  avvrYh=abs(np.mean((indata_cop[indata_cop[:,15]==10])[:,19]))
  avvrZh=abs(np.mean((indata_cop[indata_cop[:,15]==1])[:,19]))
  avvrDh=np.mean((indata_cop[indata_cop[:,20]==405])[:,19])

  avvrIxX=abs(np.mean((indata_cop[(indata_cop[:,15]==100) | (indata_cop[:,15]==404)])[:,12]))
  avvrIxY=abs(np.mean((indata_cop[indata_cop[:,15]==10])[:,12]))
  avvrIxZ=abs(np.mean((indata_cop[indata_cop[:,15]==1])[:,12]))
  avvrIxD=np.mean((indata_cop[indata_cop[:,20]==405])[:,12])

  avvrIyX=abs(np.mean((indata_cop[(indata_cop[:,15]==100) | (indata_cop[:,15]==404)])[:,13]))
  avvrIyY=abs(np.mean((indata_cop[indata_cop[:,15]==10])[:,13]))
  avvrIyZ=abs(np.mean((indata_cop[indata_cop[:,15]==1])[:,13]))
  avvrIyD=np.mean((indata_cop[indata_cop[:,20]==405])[:,13])

  avvrJ0X=abs(np.mean((indata_cop[((indata_cop[:,15]==100) | (indata_cop[:,15]==404)) & (indata_cop[:,20]==0)])[:,14]))
  avvrJ0Y=abs(np.mean((indata_cop[indata_cop[:,15]==10])[:,14]))
  avvrJ0Z=abs(np.mean((indata_cop[indata_cop[:,15]==1])[:,14]))
  avvrJ0D=np.mean((indata_cop[indata_cop[:,20]==405])[:,14])

  BeamAx=abs(np.mean((indata_cop[((indata_cop[:,15]==100) | (indata_cop[:,15]==404)) & (indata_cop[:,20]==0)])[:,18]))
  BeamAy=abs(np.mean((indata_cop[indata_cop[:,15]==10])[:,18]))
  ColumnA=abs(np.mean((indata_cop[indata_cop[:,15]==1])[:,18]))
  BeamAd=np.mean((indata_cop[indata_cop[:,20]==405])[:,18])

  No_rot_points=np.array(notrot)
  wall_points=np.vstack(((indata_cop[indata_cop[:,15]==300])[:, 3:6], (indata_cop[indata_cop[:,15]==300])[:, 6:9]))
  Rest_of_the_points=np.vstack(((indata_cop[(indata_cop[:,15]==100) | (indata_cop[:,15]==10) | (indata_cop[:,15]==1)])[:, 3:6], (indata_cop[(indata_cop[:,15]==100) | (indata_cop[:,15]==10) | (indata_cop[:,15]==1)])[:, 6:9]))

  
  momentx=[avvrIxX, avvrIyX, avvrJ0X]
  momenty=[avvrIxY, avvrIyY, avvrJ0Y]
  momentz=[avvrIxZ, avvrIyZ, avvrJ0Z]
  momentd=[avvrIxD, avvrIyD, avvrJ0D]

  numBayX=len(xrang)-1
  numBayY=len(yrang)-1
  numFloor=len(zrang)-1

  bayWidthX=xrang.tolist()
  bayWidthY=yrang.tolist()
  storyHeights=zrang.tolist()

  ceilingV=abs((xrang[-1]-xrang[0])*(yrang[-1]-yrang[0])*(avvrXw+avvrYw)/2)
  StoryV=abs((xrang[-1]-xrang[0])*(yrang[-1]-yrang[0])*((zrang.max()-zrang.min())/len(zrang)))
  Storysgm=abs((xrang[-1]-xrang[0])*(yrang[-1]-yrang[0]))

  Vnotmax=abs((xrang.shape[0]*yrang.shape[0]*avvrZl*avvrZw*avvrZh)+(xrang.shape[0]*(yrang.shape[0]-1)*avvrXw*avvrXl*avvrXh)+(yrang.shape[0]*(xrang.shape[0]-1)*avvrYw*avvrYl*avvrYh))
  Vmax=abs((xrang.shape[0]*(yrang.shape[0]-1)*avvrXw*avvrXl*avvrXh)+(yrang.shape[0]*(xrang.shape[0]-1)*avvrYw*avvrYl*avvrYh))
  density=mat_prop[2]
  if www==1:
      massX=abs(((density*(ceilingV+(Vnotmax)))))+abs(StoryV*mat_prop[3])+abs(Storysgm*mat_prop[4]) + (4790)*(xrang.max()-xrang.min())*(yrang.max()-yrang.min())/9.8 # 4790 N/m2, it is considered all as public room (worse case), /9.8 to be mass + 4790 stairs
      massL=abs(((density*(ceilingV+(Vmax)))))+abs(StoryV*mat_prop[3]) + (960)*(xrang.max()-xrang.min())*(yrang.max()-yrang.min())/9.8 # 960 N/m2 Ordinary roof
      massL=massL/(len(yrang)*len(xrang))
      massX=massX/(len(yrang)*len(xrang))
  if www==2:
      Curve=np.array(Curve)
      v_elm=np.sum(Curve[0, :, 0]*Curve[0, :, 1]*Curve[0, :, 19])
      Rest_of_the_points=Curve[0, :, 3:9]
      BeamAx=Curve[0, :, 18]
      momentx=Curve[0, :, 12:15]
      massX=density*v_elm
      massL=960*(xrang.max()-xrang.min())*(yrang.max()-yrang.min())/(len(Curve)+2) # +2 as always two at the end and start are not included, # 960 N/m2 Ordinary roof, no 9.8 as it is the load
  if www==0:
      choosenElm=indata[(indata[:,20]==405) & (np.abs(indata[:,5]-zrang.max())<0.2) & (np.abs(indata[:,8]-zrang.max())<0.2)]
      v_elm=choosenElm[0, 0]*choosenElm[0, 1]*choosenElm[0, 19]
      Rest_of_the_points=choosenElm[0, 3:9]
      BeamAx=choosenElm[0, 18]
      momentx=choosenElm[0, 12:15]
      massX=density*v_elm
      massL=960*(xrang.max()-xrang.min())*(yrang.max()-yrang.min())/(len(choosenElm)+2) #  +2 as always two at the end and start are not included, # 960 N/m2 Ordinary roof
  ini_L=((xrang[-1]-xrang[0])**2+(yrang[-1]-yrang[0])**2)*massL/12
  ini_X=((xrang[-1]-xrang[0])**2+(yrang[-1]-yrang[0])**2)*massX/12
  return numBayX, numBayY, numFloor, bayWidthX, bayWidthY, storyHeights, E, G, massX, massL, BeamAx, BeamAy, ColumnA, BeamAd, momentx, momenty, momentz, momentd, No_rot_points, wall_points, Rest_of_the_points, ini_L, ini_X


def plot_opensees(eigenValues, Load1, Load2):
    # Plot the results  
    eig=eigenValues
    plt.figure(figsize=(10, 10))
    plt.plot(np.arange(1, len(eig) + 1), np.sqrt(eig), marker='o', c="black")

    if Load1!=Load2:
        plt.title(f'Natural Frequencies under {np.round(Load1/1000)} Ton at stories and {np.round(Load2/1000)} Ton at the roof')
    else:
        plt.title(f'Natural Frequencies under {np.round(Load1/1000)} Ton at each floor')
    plt.xlabel('Mode Number')
    plt.ylabel('Frequency (Hz)')
    plt.show()

    opsv.plot_model(node_labels=0, element_labels=0)
    opsv.plot_model(node_labels=1, element_labels=0)
    opsv.plot_model(node_labels=0, element_labels=1)

    # Set figure size and font properties globally
    plt.rcParams.update({
        "figure.figsize": (5, 5),  # Width, Height in inches
        "font.size": 12,            # Font size
        "font.family": 'Times New Roman'    # Font family      
    })
    # Now when you call opsv.plot_mode_shape, it will use these settings
    for i in range(len(eig)):
        opsv.plot_mode_shape(i + 1)
        plt.xlabel('X(m)', labelpad=15)
        plt.ylabel('Y(m)', labelpad=15)
        plt.title(f'Mode Shape {i+1}')
        # Create a custom legend outside the plot
        legend_labels = ['Element', 'Node', 'Ground', 'Original Form']
        legend_markers = [
        plt.Line2D([0], [0], color='b', lw=2),  # Line for Element
        plt.Line2D([0], [0], marker='s', color='r', markersize=8, linestyle='None'),  # Square for Node
        plt.Line2D([0], [0], marker='s', color='m', markersize=8, linestyle='None'),   # Square for Ground
        plt.Line2D([0], [0], marker='o', color='g', markersize=5, linestyle='None')   # Square for Ground
        ]
        plt.legend(legend_markers, legend_labels, loc='upper left', bbox_to_anchor=(1.2, 1))
    plt.show()

# Define the rigid diaphragm with a central master point
def create_rigid_diaphragm(zLoc, nodeTags, maxtag):
    # Calculate the centroid of the nodes
    x_coords = [ops.nodeCoord(tag, 1) for tag in nodeTags]
    y_coords = [ops.nodeCoord(tag, 2) for tag in nodeTags]
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)
    
    # Create a master node at the centroid
    master_node_tag = maxtag + 1  # Ensure a unique node tag
    ops.node(int(master_node_tag), centroid_x, centroid_y, zLoc)
    ops.fix(int(master_node_tag), 0, 0, 1, 1, 1, 0)
    
    # Define the rigid diaphragm
    ops.rigidDiaphragm(3, master_node_tag, *nodeTags)
    
    return master_node_tag


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def calculate_deflection(w, L, E, I, x):
    """
    Calculate the deflection of a simply supported beam under a uniform load at a given point x using the double integration method.

    Parameters:
    w (float): Uniform load (N/m)
    L (float): Length of the beam (m)
    E (float): Young's modulus of the material (Pa)
    I (float): Moment of inertia about the axis of bending (m^4)
    x (float): Position along the length of the beam (m)

    Returns:
    float: Deflection at point x (m)
    """

    return (w * x / (24 * E * I)) * (L**3 - 2 * L * x**2 + x**3)


def compute_deformation(wei, whatT, massX, massL, Rest_of_the_points, E=12300000000.0):
    Rest_of_the_points = np.array(Rest_of_the_points)
    if whatT == 2:

        points = np.round(np.array([
            Rest_of_the_points[0, 0:3].tolist(),
            Rest_of_the_points[0, 3:6].tolist(),
            Rest_of_the_points[1, 0:3].tolist(),
            Rest_of_the_points[1, 3:6].tolist()
        ]), 1)

        # Separate the coordinates
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # Create a parameter t for interpolation
        t = np.linspace(0, 1, len(points))

        # Interpolate the curve using cubic splines
        cs_x = CubicSpline(t, x)
        cs_y = CubicSpline(t, y)
        cs_z = CubicSpline(t, z)

        # Generate points along the curve
        t_fine = np.linspace(0, 1, 100)
        x_fine = cs_x(t_fine)
        y_fine = cs_y(t_fine)
        z_fine = cs_z(t_fine)

        # Compute derivatives (tangent vectors)
        dx = cs_x(t_fine, 1)
        dy = cs_y(t_fine, 1)
        dz = cs_z(t_fine, 1)

        # Compute the magnitude of the tangent vectors
        magnitude = np.sqrt(dx**2 + dz**2)

        # Compute sine and cosine of the angles
        sin_theta = np.abs(dz / magnitude)
        cos_theta = np.abs(dx / magnitude)

        # Calculate the length of the curve
        L = np.sum(np.sqrt(np.diff(x_fine)**2 + np.diff(y_fine)**2 + np.diff(z_fine)**2))

        # Define material properties and load
        w = -(massX * 9.8 + massL * wei)/L  # Uniform load in N/m
        I_ini = np.array([[0.09957778, 0.32239408, 0.42197186],
                          [0.09282333, 0.32492761, 0.41775095],
                          [0.13013756, 0.38327528, 0.51341284]])
        
        # Interpolate moment of inertia for each xi
        xi_values = np.linspace(0, L, 100)
        I_values = np.interp(xi_values, [L/4, L/2], [(I_ini[0, 1]+I_ini[0, 2])/2, I_ini[1, 1]])

        # Calculate deflection at each point along the curve
        deflections = [calculate_deflection(w, L, E, I_values[i], xi) for i, xi in enumerate(np.linspace(0, L, 100))]

        deflectionsz = [calculate_deflection(w, L, E, I_values[i], xi*cos_theta[i]*cos_theta[i]) for i, xi in enumerate(np.linspace(0, L, 100))]

        deflections=deflectionsz

        asr = [max(x_fine) - min(x_fine), max(y_fine) - min(y_fine)]
        # Plot the deformation
        mode = np.where(np.array(asr) < 0.5)[0]
        if mode == 1:
            fig = plt.figure(figsize=(5, 15))
            ax = fig.add_subplot(311)
            ax.plot(x_fine, z_fine, label='Fitted - Original Form', c='green')
            ax.plot(x_fine, z_fine + deflections, label='Deformed Element', linestyle='--', c='red')
            ax.set_xlabel('X(m)')
            ax.set_ylabel('Z(m)')
            ax.set_title(f'Under {np.round(-L*w/1000, 1)} kN or {np.round(-w/1000, 1)} kN/m Uniform Load')
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            ax.set_ylim(ymin=4, ymax=8)
            ax = fig.add_subplot(312)
            ax.plot(xi_values, deflections, label='Deformation', linestyle='--', c='red')
            ax.set_xlabel('X(m)')
            ax.set_ylabel('Z(m)')
            ax.set_ylim(ymin=-1.6, ymax=0)
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.show()
        if mode==0:
            fig = plt.figure(figsize=(5,7))
            ax = fig.add_subplot(311)
            ax.plot(y_fine, z_fine, label='Original Form', c='green')
            # ax.scatter(y, z, color='black', label='Original Points')
            ax.plot(y_fine, z_fine+deflections, label='Deformed Element', linestyle='--', c='red')
            ax.set_xlabel('Y(m)')
            ax.set_ylabel('Z(m)')
            ax.set_ylim(ymin=6, ymax=10)
            ax.set_title(f'Under {np.round(-L*w/1000, 1)} kN or {np.round(-w/1000, 1)} kN/m Uniform Load')
            # ax.set_aspect(asr[2]/asr[0], adjustable='box')
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            ax = fig.add_subplot(312)
            ax.plot(y_fine, deflections, label='Deformation', linestyle='--', c='red')
            ax.set_xlabel('Y(m)')
            ax.set_ylabel('Z(m)')
            ax.set_ylim(ymin=-0.06, ymax=0)
            # ax.set_aspect(asr[2]/asr[0], adjustable='box')
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.show()
    if whatT==0:
        # Define the points
        points = np.round(np.array([
            Rest_of_the_points[0:3],
            Rest_of_the_points[3:6]
        ]), 1)

        # Separate the coordinates
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # Create a parameter t for interpolation
        t = np.linspace(0, 1, len(points))

        # Interpolate the curve using cubic splines
        cs_x = CubicSpline(t, x)
        cs_y = CubicSpline(t, y)
        cs_z = CubicSpline(t, z)

        # Generate points along the curve
        t_fine = np.linspace(0, 1, 100)
        x_fine = cs_x(t_fine)
        y_fine = cs_y(t_fine)
        z_fine = cs_z(t_fine)

        # Calculate the length of the curve
        L = np.sum(np.sqrt(np.diff(x_fine)**2 + np.diff(y_fine)**2 + np.diff(z_fine)**2))

        # Define material properties and load
        w = -(massX*9.8+massL*wei)/L # Uniform load in N/m
        # E = 12.3e9  # Young's modulus in Pascals (e.g., steel)
        I_ini=np.array([0.02344381, 0.06927005, 0.09271385])
        I = I_ini[1]  # Moment of inertia in m^4
        # Calculate deflection at each point along the curve
        deflections = [calculate_deflection(w, L, E, I, xi) for xi in np.linspace(0, L, 100)]
        asr=[max(x_fine)-min(x_fine), max(y_fine)-min(y_fine)]
        # Plot the deformation
        mode=np.where(np.array(asr)<0.5)[0]
        if mode==0:
            fig = plt.figure(figsize=(5,7))
            ax = fig.add_subplot(311)
            ax.plot(y_fine, z_fine, label='Original Form', c='green')
            # ax.scatter(y, z, color='black', label='Original Points')
            ax.plot(y_fine, z_fine+deflections, label='Deformed Element', linestyle='--', c='red')
            ax.set_xlabel('Y(m)')
            ax.set_ylabel('Z(m)')
            ax.set_ylim(ymin=6, ymax=10)
            ax.set_title(f'Under {np.round(-L*w/1000, 1)} kN or {np.round(-w/1000, 1)} kN/m Uniform Load')
            # ax.set_aspect(asr[2]/asr[0], adjustable='box')
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            ax = fig.add_subplot(312)
            ax.plot(y_fine, deflections, label='Deformation', linestyle='--', c='red')
            ax.set_xlabel('Y(m)')
            ax.set_ylabel('Z(m)')
            ax.set_ylim(ymin=-0.06, ymax=0)
            # ax.set_aspect(asr[2]/asr[0], adjustable='box')
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.show()
        if mode == 1:
            fig = plt.figure(figsize=(5, 15))
            ax = fig.add_subplot(311)
            ax.plot(x_fine, z_fine, label='Fitted - Original Form', c='green')
            ax.plot(x_fine, z_fine + deflections, label='Deformed Element', linestyle='--', c='red')
            ax.set_xlabel('X(m)')
            ax.set_ylabel('Z(m)')
            ax.set_title(f'Under {np.round(-L*w/1000, 1)} kN or {np.round(-w/1000, 1)} kN/m Uniform Load')
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            ax.set_ylim(ymin=4, ymax=8)
            ax = fig.add_subplot(312)
            ax.plot(xi_values, deflections, label='Deformation', linestyle='--', c='red')
            ax.set_xlabel('X(m)')
            ax.set_ylabel('Z(m)')
            ax.set_ylim(ymin=-1.6, ymax=0)
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.show()
    return deflections

def OpenSees_M(numEigen, numBayX, numBayY, numFloor, bayWidthX, bayWidthY, storyHeights, E, G, massX, massL, BeamAx, BeamAy, ColumnA, BeamAd, momentx, momenty, momentz, momentd, No_rot_points, wall_points, Rest_of_the_points, whatT, wei, new_wand, mode, myelements, mynodes, ini_L, ini_X, modeL):
    # set some properties
    ops.wipe()
    ops.model('Basic', '-ndm', 3, '-ndf', 6)
    '''properties
    # units N, kg, m'''
    M = 0.
    coordTransf = "Linear"  # Linear, PDelta, Corotational
    massType = "-cMass"  # -lMass, -cMass
    if whatT==1:
        if mode==0:
            nodeTag = 1
            # Store nodes at each zLoc
            nodes_at_zLoc = {}

            # add the nodes
            #  - floor at a time
            ywidth=np.diff(bayWidthY)
            xwidth=np.diff(bayWidthX)
            zwidth=np.diff(storyHeights)  
            zLoc = storyHeights[0]
            for k in range(0, numFloor + 1):
                xLoc = bayWidthX[0] 
                for i in range(0, numBayX + 1):
                    yLoc = bayWidthY[0]
                    for j in range(0, numBayY + 1):
                        ops.node(nodeTag, xLoc, yLoc, zLoc)
                        if k==numFloor:
                            ops.mass(nodeTag, massL, massL, 0.01, 1.0e-10, 1.0e-10, 1.0e-10)
                        else:
                            ops.mass(nodeTag, massX, massX, 0.01, 1.0e-10, 1.0e-10, 1.0e-10)
                        if k == 0:
                            ops.fix(nodeTag, 1, 1, 1, 1, 1, 1)
                        if k!=0:
                            # Store node tags by zLoc
                            if zLoc not in nodes_at_zLoc:
                                nodes_at_zLoc[zLoc] = []
                            nodes_at_zLoc[zLoc].append(nodeTag)
                            
                        if j < numBayY: yLoc += ywidth[j]
                        nodeTag += 1
                        
                    if i < numBayX: xLoc += xwidth[i]

                if k < numFloor: zLoc += zwidth[k]
            # # Create rigid diaphragms for each zLoc
            current_max_tag=nodeTag
            for zLoc, nodeTags in nodes_at_zLoc.items():
                current_max_tag = create_rigid_diaphragm(zLoc, nodeTags, current_max_tag)

            # add column element Z
            ops.geomTransf(coordTransf, 1, 1, 0, 0)
            ops.geomTransf(coordTransf, 2, 0, 0, 1)

            eleTag = 1
            nodeTag1 = 1

            for k in range(0, numFloor):
                for i in range(0, numBayX+1):
                    for j in range(0, numBayY+1):
                        nodeTag2 = nodeTag1 + (numBayX+1)*(numBayY+1)
                        iNode = ops.nodeCoord(nodeTag1)
                        jNode = ops.nodeCoord(nodeTag2)
                        ops.element('elasticBeamColumn', eleTag, nodeTag1, nodeTag2, ColumnA, E, G, momentz[2], momentz[0], momentz[1], 1, '-mass', M, massType)
                        eleTag += 1
                        nodeTag1 += 1


            nodeTag1 = 1+ (numBayX+1)*(numBayY+1)
            #add beam elements X
            for j in range(1, numFloor + 1):
                for i in range(0, numBayX):
                    for k in range(0, numBayY+1):
                        nodeTag2 = nodeTag1 + (numBayY+1)
                        iNode = ops.nodeCoord(nodeTag1)
                        jNode = ops.nodeCoord(nodeTag2)
                        ops.element('elasticBeamColumn', eleTag, nodeTag1, nodeTag2, BeamAx, E, G, momentx[2], momentx[0], momentx[1], 2, '-mass', M, massType)
                        eleTag += 1
                        nodeTag1 += 1
                    
                nodeTag1 += (numBayY+1)

            nodeTag1 = 1+ (numBayX+1)*(numBayY+1)
            #add beam elements Y
            for j in range(1, numFloor + 1):
                for i in range(0, numBayY+1):
                    for k in range(0, numBayX):
                        nodeTag2 = nodeTag1 + 1
                        iNode = ops.nodeCoord(nodeTag1)
                        jNode = ops.nodeCoord(nodeTag2)
                        ops.element('elasticBeamColumn', eleTag, nodeTag1, nodeTag2, BeamAy, E, G, momenty[2], momenty[0], momenty[1], 2, '-mass', M, massType)
                        eleTag += 1
                        nodeTag1 += 1
                    nodeTag1 += 1
        if mode==1:
            # Define your nodes and elements here
            nodes = mynodes  
            elements = myelements

            # Create node
            for i in range(len(nodes)):
                coords=nodes[i, :]
                ops.node(int(i+1), *coords[0:3])
                if abs(mynodes[i,2]-np.max(mynodes[i,2]))<0.36:
                    masss=massL
                    inertia=ini_L
                    ops.mass(int(i+1), masss, masss, 0.0, 0.0, 0.0, inertia)
                else: 
                    masss=massX
                    inertia=ini_X
                    ops.mass(int(i+1), masss, masss, 0.0, 0.0, 0.0, inertia)

            # Fix nodes
            for i in range(len(nodes)):
                if mynodes[i,2]<0.5:
                    ops.fix(i+1, 1, 1, 1, 1, 1, 1)  # All ground nodes are fixed in 3D

            
            znodes=[]
            nxt=(numBayX+1)*(numBayY+1)
            for ki in range(numFloor):
                xynodes=[]
                for kj in range((numBayX+1)*(numBayY+1)):
                    xynodes.append(nodes[kj+nxt,3])
                nxt=nxt+((numBayX+1)*(numBayY+1))
                znodes.append(xynodes)

            nii=(numBayX+1)*(numBayY+1)*(numFloor+1)+100
            list_nii=[]
            for nnn in znodes:
                dumy=[]
                for ijij in nnn:
                    dumy.append(nodes[int(ijij-1), 0:3])
                meann=np.mean(np.array(dumy)[0:2], axis=0)
                ops.node(int(nii), *[meann[0], meann[1], nodes[int(ijij-1),2]])
                ops.fix(int(nii), 0, 0, 1, 1, 1, 0)
                for kk in nnn:
                    ops.rigidDiaphragm (3, int(nii), int(kk))
                list_nii.append(nii)
                nii=nii+1


            #geomTransf
            if modeL==0:
                ops.geomTransf ('Linear ', 3 ,0 ,0 ,1)
                ops.geomTransf ('Linear ', 1 ,0 ,1 ,0)
                ops.geomTransf ('Linear ', 2 ,1 ,0 ,0)

            if modeL==1:
                ops.geomTransf ('PDelta  ', 3 ,0 ,0 ,1)
                ops.geomTransf ('PDelta  ', 1 ,0 ,1 ,0)
                ops.geomTransf ('PDelta  ', 2 ,1 ,0 ,0)

            for i in range(len(elements)):
                Area = elements[i, 18]
                Jxx = elements[i, 14]
                Iy = elements[i, 12]
                Iz = elements[i, 13]
                ops.section('Elastic', i+1, E, Area, Iz, Iy, G, Jxx)

            for i in range(len(elements)):

                start=int(elements[i,21]+1)
                end=int(elements[i,22]+1)
                
                # Define the element

                if modeL==0:
                    if elements[i,15]==100:
                        ops.element('elasticBeamColumn', i+1, start, end, i+1, 3)
                    if elements[i,15]==10:
                        ops.element('elasticBeamColumn', i+1, start, end, i+1, 2)
                    if elements[i,15]==1:
                        ops.element('elasticBeamColumn', i+1, start, end, i+1, 1)

                if modeL==1:
                    if elements[i,15]==100:
                        ops.element('nonlinearBeamColumn  ', i+1, start, end, 6, i+1, 3)
                    if elements[i,15]==10:
                        ops.element('nonlinearBeamColumn  ', i+1, start, end, 6, i+1, 2)
                    if elements[i,15]==1:
                        ops.element('nonlinearBeamColumn  ', i+1, start, end, 6, i+1, 1)
        # calculate eigenvalues & print results
        eigenValues = ops.eigen(numEigen)
        print('Natural frequencies: ', np.sqrt(np.array(eigenValues)))
        plot_opensees(eigenValues, massX, massL)
        return eigenValues

    else:
        return compute_deformation(wei, whatT, massX, massL, Rest_of_the_points, E)
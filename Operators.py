"""
Disclaimer:

This code is designed with several simplifications to enhance its usability and performance. While it has been tested and utilized in published research, it may not cover all potential scenarios and edge cases. Users should evaluate and validate the results based on their specific needs and requirements. The code is provided to support further research and development, and contributions for improvements are encouraged.

"""

# Define all Functions

import open3d as o3d
import numpy as np
import scipy.spatial as spatial
from numpy import *
from openseespy.opensees import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Define fonts
font = {'weight' : 'bold',
        'size'   : 10}

plt.rc('font', **font)

def points_within_range(points, min_range, max_range, indexs):
    """
    Find points within a given range using vectorization.

    Parameters:
    points (numpy array): Array of points with shape (n, 2).
    min_range (float): Minimum range value.
    max_range (float): Maximum range value.

    Returns:
    numpy array: Points within the given range.
    """
    mask = (points[:, indexs] >= min_range) & (points[:, indexs] <= max_range)
    return points[mask]

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def compute_and_plot_histogram_with_peaks(perc, points, whatTT, bin_size=50, threshold=0.1, plot_kon=False):
    """
    Computes and plots the histogram of Z values in a point cloud with specified bin size,
    and identifies and plots the peaks in the histogram.

    Parameters:
    points (numpy.ndarray): The point cloud data as a NumPy array with shape (N, 3).
    bin_size (int): The size of each bin for the histogram.

    Returns:
    hist (numpy.ndarray): The histogram counts for each bin.
    bin_edges (numpy.ndarray): The edges of the bins.
    peaks (numpy.ndarray): The indices of the peaks in the histogram.
    """
    # Extract the Z values
    values = points[:, whatTT]
    values_num=len(values)

    # Determine the range of Z values
    min_z = np.min(values)
    max_z = np.max(values)

    # Create bins of specified size
    bins = np.arange(min_z- bin_size, max_z + 2*bin_size, bin_size)

    # Compute the histogram
    hist, bin_edges = np.histogram(values, bins=bins)

    hist_not_norm=hist.copy()

    # Normalize the histogram
    hist = hist / values_num

    # Calculate the width of each bin
    bin_widths = np.diff(bin_edges)

    # Find peaks in the histogram
    peaks, properties = find_peaks(hist)

    # Get the Z values at the peaks
    peak_values = bin_edges[peaks]

    # Find the percentile of the histogram counts
    percentilev = np.percentile(hist, perc)

    # Filter peaks that are greater than the percentile
    high_peaks = peaks[hist[peaks] > percentilev]
    high_peak_values = bin_edges[high_peaks]

    # Get the widths of the bins for the high peaks
    high_peak_widths = bin_widths[high_peaks]

    # Calculate the ranges of values close to high_peak_values
    close_value_ranges = [(value - threshold, value + threshold) for value in high_peak_values]

    if plot_kon:
        # Plot the histogram using plt.bar
        nam_nam = ['X', 'Y', 'Z']
        plt.figure(figsize=(10, 6))
        plt.bar(bin_edges[:-1], hist, width=bin_widths, edgecolor='black', alpha=0.7, label='Histogram')
        plt.plot(bin_edges[high_peaks], hist[high_peaks], 'ro', label='Peaks')
        plt.xlabel('{0} value'.format(nam_nam[whatTT]))
        plt.ylabel('Frequency')
        plt.title('Histogram of {0} values in Point Cloud with Peaks'.format(nam_nam[whatTT]))
        plt.legend(bbox_to_anchor=(1.2, 1.0), loc='upper left')
        plt.grid(True)
        plt.show()

    return hist_not_norm, bin_edges, peaks, peak_values, high_peak_values, close_value_ranges, high_peak_widths, high_peaks, hist[high_peaks]


def discretize_line(start, end, num_points):
    return np.linspace(start, end, num_points)

# Function to find the index of a node by its id
def find_index_by_id(points2, id):
    return np.where(points2[:, 3] == id)[0][0]

from matplotlib.lines import Line2D

def segment_plotter(input_pcd, indicate_xyz, X, Y, Z, thrsh, text, asr, size=(10 ,10)):
    # Sample 3D point cloud
    point_cloud = input_pcd

    # List of numbers
    if indicate_xyz == 0: numbers = X
    if indicate_xyz == 1: numbers = Y
    if indicate_xyz == 2: numbers = Z

    # Define colors
    colors = np.linspace(0.1, 1, len(numbers))

    # Create an array to store the points with their corresponding colors
    colored_points = np.asarray([[0, 0, 0, 0], [0, 0, 0, 0]])
    colored_points_non = np.asarray([[0, 0, 0, 0], [0, 0, 0, 0]])

    # Assign colors based on proximity to numbers
    for i, num in enumerate(numbers):
        mask = np.where(np.abs(point_cloud[:, indicate_xyz] - num) < thrsh)[0]  # Adjust threshold as needed
        masked_points = point_cloud[mask]
        masked_colors = np.ones((masked_points.shape[0], 1)) * colors[i]
        colored_points = np.append(colored_points, np.hstack((masked_points, masked_colors)), axis=0)
        non_masked_points = point_cloud[~mask]
        non_masked_colors = np.ones((non_masked_points.shape[0], 1)) * 0
        colored_points_non = np.append(colored_points_non, np.hstack((non_masked_points, non_masked_colors)), axis=0)

    # Get the coordinates of the skeleton points

    # Create a 3D plot
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection='3d')

    datap = colored_points_non[2:-1, :]
    x = datap[:, 0]
    y = datap[:, 1]
    z = datap[:, 2]
    ax.scatter(x, y, z, zdir='z', c='black', alpha=0.1, label="Other points")

    datap = colored_points[2:-1, :]
    x = datap[:, 0]
    y = datap[:, 1]
    z = datap[:, 2]
    colors2 = datap[:, 3]

    ax.scatter(x, y, z, zdir='z', c=colors2)

    # Set labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_box_aspect(asr)

    # Set title
    ax.set_title(text)

    # Create custom legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Segment {i+1}',
                              markerfacecolor=plt.cm.viridis(c), markersize=10)
                      for i, c in enumerate(colors)]

    # Add the extra element for 'Other points'
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Other points',
                                  markerfacecolor='black', markersize=10))

    # Create the legend
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.2, 1.0), loc='upper left')

    # Show the plot
    plt.show()

    return colored_points[2:-1, :]

def remove_outliers(data, threshold, mod=0):
    """
    Remove outliers from a NumPy array using percentile thresholds.

    Parameters:
    data (numpy.ndarray): The input array.
    lower_percentile (float): The lower percentile threshold.
    upper_percentile (float): The upper percentile threshold.

    Returns:
    numpy.ndarray: The array with outliers removed.
    """
    if data.size == 0:
        return data
    upper_percentile=100-threshold
    lower_percentile=threshold
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    if mod==0:
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    if mod==1:
        filtered_data = data[(data >= lower_bound)]
    return filtered_data

def magnitude_order_array(arr):
    abs_arr = np.abs(arr)
    order_arr = np.log10(abs_arr)
    res_arr = np.floor(order_arr)
    return res_arr.astype(int)

def arr_to_o3d(arrIn):
  pcd22=arrIn
  pcd22=np.array(pcd22)
  pcd_o3dfiltered = o3d.geometry.PointCloud()  # create point cloud object
  pcd_o3dfiltered.points = o3d.utility.Vector3dVector(pcd22)  # set pcd_np as the point cloud point
  return pcd_o3dfiltered

def arr_to_o3d(arrIn):
  pcd22=arrIn
  pcd22=np.array(pcd22)
  pcd_o3dfiltered = o3d.geometry.PointCloud()  # create point cloud object
  pcd_o3dfiltered.points = o3d.utility.Vector3dVector(pcd22)  # set pcd_np as the point cloud point
  return pcd_o3dfiltered

import numpy as np
from scipy.spatial import KDTree
import concurrent.futures

def estimate_normals(point_cloud, k=10):
  point_cloud=np.asarray(point_cloud.points)
  # Build a k-d tree
  tree = KDTree(point_cloud)

  # Initialize an array to hold the normals
  normals = np.zeros(point_cloud.shape)

  for i, point in enumerate(point_cloud):
      # Find the k nearest neighbors
      dists, idxs = tree.query(point, k)

      # Perform PCA to find the normal
      cov = np.cov(point_cloud[idxs].T)
      w, v = np.linalg.eig(cov)
      normal = v[:, np.argmin(w)]

      # Ensure the normal is pointing in the right direction
      if np.dot(normal, point) < 0:
          normal = -normal

      normals[i] = normal

  return normals

def remove_points_using_normals(pcd, threshold, normals, xxx):

  # Get indices of points with normals that are not vertical
  indices = np.where(abs(normals[:, xxx]) > threshold)[0]

  # Select points with normals that are not vertical
  points = np.asarray(pcd.points)[indices, :]
  normals = normals[indices, :]

  # Create a new point cloud object
  pcd_ground_removed = o3d.geometry.PointCloud()

  # Assign the points and normals to the new point cloud object
  pcd_ground_removed.points = o3d.utility.Vector3dVector(points)
  pcd_ground_removed.normals = o3d.utility.Vector3dVector(normals)

  return pcd_ground_removed



from sklearn.neighbors import NearestNeighbors

def remove_sparse_points(point_cloud, radius, min_neighbors):
    """
    Remove sparse points from a point cloud.

    Parameters:
    point_cloud (numpy.ndarray): The point cloud, shape (n_points, n_dims).
    radius (float): The radius within which to count neighbors.
    min_neighbors (int): The minimum number of neighbors a point must have to be kept.

    Returns:
    numpy.ndarray: The filtered point cloud, shape (n_kept_points, n_dims).
    """
    point_cloud=np.asarray(point_cloud.points)
    nbrs = NearestNeighbors(radius=radius, algorithm='auto').fit(point_cloud)
    distances, indices = nbrs.radius_neighbors(point_cloud)
    mask = np.array([len(neighbors) >= min_neighbors for neighbors in indices])
    return point_cloud[mask]

from sklearn.cluster import KMeans, DBSCAN, OPTICS, HDBSCAN
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as hcluster

def clustering(param, dataIn, mode=1, eps=0.15, scalme=False):
  # Normalisation:
  if scalme: scaled_points = StandardScaler().fit_transform(dataIn)
  # Clustering:
  if mode==3:
    thresh = param
    clusters = hcluster.fclusterdata(dataIn, thresh, criterion="distance")
    output=clusters
  if mode==2:model = DBSCAN(eps=eps, min_samples=param)
  if mode==1: model = HDBSCAN(min_cluster_size=param)
  if mode==4: model = OPTICS(min_samples=param)
  if mode==5: model = KMeans(n_clusters=param)
  if not mode==3:
    if not scalme: model.fit(dataIn)
    if scalme: model.fit(scaled_points)
    output=model.labels_
  return output

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plotter2(datap, text, mode, colors, hMany, asr, size=(10, 10)):
    arr_min = np.min(colors)
    arr_max = np.max(colors)
    colors = (colors - arr_min) / (arr_max - arr_min)
    # Get the coordinates of the skeleton points
    x = datap[:, 0]
    y = datap[:, 1]
    z = datap[:, 2]

    # Create a 3D plot
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection='3d')

    if mode == 1:
        ax.scatter(x, y, z, zdir='z', c=colors, label=text)
        ax.legend(bbox_to_anchor=(1.2, 1.0), loc='upper left')
    if mode == 0:
        ax.scatter(x, y, z, zdir='z', label=text)
        ax.legend(bbox_to_anchor=(1.2, 1.0), loc='upper left')

    if mode == 2:
        # Define colors
        text_colors = np.min(colors)
        text_colors_list = []
        for i in range(hMany):
            text_colors_list.append(text_colors)
            text_colors += 1
        text_colors_list=np.asarray(text_colors_list)
        text_colors_list = (text_colors_list - arr_min) / (arr_max - arr_min)

        # Create a 3D plot
        ax.scatter(x, y, z, zdir='z', c=colors)

        # Create custom legend
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Segment {i+1}',
                                  markerfacecolor=plt.cm.viridis(text_colors_list[i]), markersize=10)
                           for i in range(hMany)]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.2, 1.0), loc='upper left')

    # Set labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_box_aspect(asr)
    # Set title
    ax.set_title(text)

    # Show the plot
    plt.show()
    return

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def project_point_onto_plane(point, point_on_plane, normal_vector):
    point = np.array(point)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    vector_from_plane = point - point_on_plane
    distance = np.dot(vector_from_plane, normal_vector)
    projection = point - distance * normal_vector
    
    # Define basis vectors
    min_index = np.argmin(np.abs(normal_vector))
    arbitrary_vector = np.zeros(3)
    arbitrary_vector[min_index] = 1
    basis_vector1 = np.cross(normal_vector, arbitrary_vector)
    basis_vector1 = basis_vector1 / np.linalg.norm(basis_vector1)
    basis_vector2 = np.cross(normal_vector, basis_vector1)
    basis_vector2 = basis_vector2 / np.linalg.norm(basis_vector2)
    
    # Calculate the 2D coordinates on the plane
    vector_in_plane = projection - point_on_plane
    new_x = np.dot(vector_in_plane, basis_vector1)
    new_y = np.dot(vector_in_plane, basis_vector2)
    new_coords = np.array([new_x, new_y])

    # Calculate the conversion scale
    original_distance = np.linalg.norm(vector_from_plane)
    projected_distance = np.linalg.norm(new_coords)
    conversion_scale = projected_distance / original_distance if original_distance != 0 else 1
    
    return projection, new_coords, conversion_scale

# Function to compute the area of the convex hull of a set of points
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull

def compute_area_and_moments(points, numI, mode, plot_it, conversion_scale):
    if len(points) < 5:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    if mode=='poly': polygon = Polygon(points).convex_hull
    if mode=='rec': polygon = Polygon(points).minimum_rotated_rectangle
    if mode=='other':
        polygon = ConvexHull(points)

    if plot_it==1 and mode!='other':
      # Ensure plotting runs regardless of any issues
      try:
          if polygon.geom_type == 'Polygon':
            xxx, yyy = polygon.exterior.xy
            xxx = abs(np.array(xxx) * conversion_scale)
            yyy = abs(np.array(yyy) * conversion_scale)
            plt.plot(xxx, yyy)
            plt.fill(xxx, yyy, alpha=0.5, fc='r', ec='black')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title("Cross Section")
            plt.show()
          else:
            plt.show()
      except Exception as e:
          print(f"An error occurred while plotting: {e}")
    if mode=='poly':
        Aa = polygon.area * (conversion_scale ** 2)

        # Calculate moments of inertia
        Ix = 0
        Iy = 0
        for i in range(len(polygon.exterior.coords) - 1):
            x1, y1 = polygon.exterior.coords[i]
            x2, y2 = polygon.exterior.coords[i + 1]
            common_term = (x1 * y2 - x2 * y1)
            Ix += (y1**2 + y1 * y2 + y2**2) * common_term
            Iy += (x1**2 + x1 * x2 + x2**2) * common_term
        Ix = abs(Ix) / 12.0
        Iy = abs(Iy) / 12.0
        Jx_y = Ix + Iy

        # Compute bounding box dimensions
        minx, miny, maxx, maxy = polygon.bounds
        length = (maxx - minx) * conversion_scale
        width = (maxy - miny) * conversion_scale

    if mode=='other':
        Aa = polygon.volume* (conversion_scale ** 2)
        centroid = np.mean(points, axis=0)
        length = (np.max(points[:, 0]) - np.min(points[:, 0]))* conversion_scale
        width = (np.max(points[:, 1]) - np.min(points[:, 1]))* conversion_scale
        Ix=(width*(length**3))/12.0
        Iy=((width**3)*length)/12.0

    if mode=='rec':
        # Compute bounding box dimensions
        minx, miny, maxx, maxy = polygon.bounds
        length = (maxx - minx) * conversion_scale
        width = (maxy - miny) * conversion_scale
        Ix=(width*(length**3))/12.0
        Iy=((width**3)*length)/12.0
        Jx_y=Ix + Iy
        Aa=length*width

    return Aa, Ix, Iy, Jx_y, length, width, polygon

def break_point_cloud(pcd_o3d, SindX, SindY, anormals, OurTH, OurCoefficient):
    # Convert point cloud to numpy array
    points = np.asarray(pcd_o3d.points)
    pcdp = points.copy()
    max_th= np.max(pcdp[:, 2])
    mean_th= np.mean(pcdp[:, 2])

    points = points[pcdp[:, 2]>= max_th-OurCoefficient*mean_th]
    anormals = anormals[pcdp[:, 2]>= max_th-OurCoefficient*mean_th]

    # Initialize a list to store the resulting point clouds
    point_clouds = []
    thexypoints = []

    # Loop through each combination of SindX and SindY
    for i in range(len(SindX)):
        for j in range(len(SindY)):
            # Define the bounds
            x_min, x_max = SindX[i] - OurTH, SindX[i] + OurTH
            y_min, y_max = SindY[j] - OurTH, SindY[j] + OurTH

            # Filter points that satisfy the conditions
            mask = (points[:, 0] > x_min) & (points[:, 0] < x_max) & (points[:, 1] > y_min) & (points[:, 1] < y_max)
            filtered_points = points[mask]
            filtered_normals = anormals[mask]

            if filtered_points.shape[0]>5:
              # Create a new point cloud from the filtered points
              pcd_filtered = o3d.geometry.PointCloud()
              pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)
              pcd_filtered.normals = o3d.utility.Vector3dVector(filtered_normals)

              # Append the filtered point cloud to the list
              point_clouds.append(pcd_filtered)
              if SindX[i]==SindX[0] : SindX[i]= np.min(pcdp[:,0])
              if SindY[j]==SindY[0] : SindY[j]= np.min(pcdp[:,1])
              thexypoints.append([SindX[i],SindY[j]])

    return point_clouds, thexypoints

def ave_xy(sorted_data, threshold):
    # Initialize variables
    sorted_data.sort()
    groups = []
    current_group = [sorted_data[0]]

    # Group similar numbers
    for i in range(1, len(sorted_data)):
        if sorted_data[i] - current_group[-1] <= threshold:
            current_group.append(sorted_data[i])
        else:
            groups.append(current_group)
            current_group = [sorted_data[i]]

    groups.append(current_group)

    # Calculate the averages
    averages = [np.mean(group) for group in groups]

    # Print the groups and their averages
    for i, group in enumerate(groups):
        print(f"Group {i+1}: {group}, Average: {averages[i]}")

    return averages

# Function to find edges with the same x, y, or z coordinates and satisfy the condition
def find_edges(nodes, lDelta, trixy):
    edges = []
    coord_diff = []
    midss22=[]
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            x1, y1, z1, id1 = nodes[i]
            x2, y2, z2, id2 = nodes[j]
            # Calculate differences
            diff = [abs(x1 - x2), abs(y1 - y2), abs(z1 - z2)]
            diff=np.array(diff)
            lDelta=np.array(lDelta)
            dist_error=np.sqrt(sum(np.square(diff)))-np.sqrt(sum(np.square(lDelta)))
            count_greater_than_0_2 = sum(1 for x in diff if x < 2)
            if not count_greater_than_0_2 == 2 and abs(dist_error) < trixy:
                edges.append([int(id1), int(id2)])
                coord_diff.append([(x1, y1, z1, id1), (x2, y2, z2, id2), (x1 - x2, y1 - y2, z1 - z2, id1-id2)])
                midss22.append([diff/2])

    return edges, coord_diff, midss22


def plot_the_pcd(pcd_o3d, asr):
   # Plot the cleaned point cloud

    pcdtocrossCC=np.asarray(pcd_o3d.voxel_down_sample(voxel_size=0.1).points).copy()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    XX=pcdtocrossCC[:,0]
    YY=pcdtocrossCC[:,1]
    ZZ=pcdtocrossCC[:,2]
    ax.scatter(XX, YY, ZZ, c='green', alpha=0.1, s=5, label="Raw Point Cloud")

    # Set labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_box_aspect(asr)

    # Set title
    ax.set_title("Raw Structure Point Cloud")
    ax.legend(bbox_to_anchor=(1.2, 1.0), loc='upper left')

    plt.show()

def remove_points(pcd, threshold, normals, xxx):
    # Get indices of points with normals that are not vertical
    non_ground_indices = np.where(abs(normals[:, xxx]) < threshold)[0]
    ground_indices = np.where(abs(normals[:, xxx]) >= 1-threshold)[0]

    # Select points with normals that are not vertical
    non_ground_points = np.asarray(pcd.points)[non_ground_indices, :]
    non_ground_normals = normals[non_ground_indices, :]

    # Select ground points
    ground_points = np.asarray(pcd.points)[ground_indices, :]
    ground_normals = normals[ground_indices, :]

    # Create new point cloud objects
    pcd_ground_removed = o3d.geometry.PointCloud()
    pcd_ground = o3d.geometry.PointCloud()

    # Assign the points and normals to the new point cloud objects
    pcd_ground_removed.points = o3d.utility.Vector3dVector(non_ground_points)
    pcd_ground_removed.normals = o3d.utility.Vector3dVector(non_ground_normals)

    pcd_ground.points = o3d.utility.Vector3dVector(ground_points)
    pcd_ground.normals = o3d.utility.Vector3dVector(ground_normals)

    return pcd_ground_removed, pcd_ground

def prepare_enormals(whatT, pcd_o3d):
   # Estimate normals
    if whatT==1 or whatT==0:
        enormals=estimate_normals(pcd_o3d, k=10)
        pcdtocross1, _ =remove_points(pcd_o3d, 0.1, enormals,2)

    if whatT==2:
        enormals=estimate_normals(pcd_o3d, k=50)
        pcdtocross1, _ =remove_points(pcd_o3d, 0.01, enormals,2)

    if whatT==1: pcdtocross=remove_sparse_points(pcdtocross1, 0.5, 30)

    if whatT==0: pcdtocross=np.asarray(pcdtocross1.points)

    if whatT==2: pcdtocross=remove_sparse_points(pcdtocross1, 0.5, 30)
    return enormals, pcdtocross

def rough_seg(enormals, whatT, pcd_o3d, asr):
   # Rough Segmentation of the Structure toward X, Y, and Z

    mayZ3=[]
    mayX3=[]
    mayY3=[]
    points_in_each_seg=[]
    names=["X", "Y", "Z"]

    for j in [0,1,2]:
        groups=[]
        what=j
        if whatT==1: pcdfiltered=remove_points_using_normals(pcd_o3d, 0.9, enormals,what)
        if whatT==0: pcdfiltered=remove_points_using_normals(pcd_o3d, 0.95, enormals,what)
        if whatT==2: pcdfiltered=remove_points_using_normals(pcd_o3d, 0.9, enormals,what)
        if whatT==1:
            pcdfiltered=remove_sparse_points(pcdfiltered, 0.5, 20)
            factor=25
        if whatT==0:
            pcdfiltered=remove_sparse_points(pcdfiltered, 0.5, 70)
            factor=35
        if whatT==2:
            pcdfiltered=remove_sparse_points(pcdfiltered, 0.5, 20)
            factor=20

        mydata=pcdfiltered
        mydata2=mydata
        mydata2[:,what]=mydata2[:,what]*factor

        if whatT==1:
            pcdfiltered2=remove_sparse_points(arr_to_o3d(mydata2), 0.5, 20)
            pnum=1000
            mod=1
        if whatT==0:
            pcdfiltered2=remove_sparse_points(arr_to_o3d(mydata2), 0.5, 20)
            pnum=600
            mod=1
        if whatT==2:
            pcdfiltered2=remove_sparse_points(arr_to_o3d(mydata2), 0.5, 30)
            # Remove tilted ceiling
            if not what==2:
                zwhind=np.where(pcdfiltered2[:,2]<np.max(pcdfiltered2[:,2])-0.15*np.mean(pcdfiltered2[:,2]))[0]
                pcdfiltered2=pcdfiltered2[zwhind,:]
            pnum=200
            mod=1

        labels=clustering(pnum, pcdfiltered2, mod, 0.5, False)
        dfl=labels
        dfl=dfl.reshape(dfl.shape[0],1)
        jai=np.min(dfl)
        jai2=jai
        how_many=abs(np.min(dfl) - np.max(dfl))+1
        raga=range(how_many)
        print("There are {0} segments toward {1} axis".format(how_many, names[j]))
        print('---------- Details ----------')

        for i in raga:
            ind2=np.where(dfl==jai)[0]
            jai=jai+1
            groups.append([len(ind2)])

        groups2=np.asarray(groups)
        groups3=magnitude_order_array(groups2)
        g_ave=np.mean(groups3)
        threshold=  int(g_ave)

        for kk in raga:
            ind=np.where(dfl==jai2)[0]
            jai2=jai2+1
            cri=magnitude_order_array(len(ind))
            # cleaned=remove_outliers(pcdfiltered2[ind,what],25, 0) # This is optional
            cleaned=pcdfiltered2[ind,what]
            print('---------- Segment {0} ----------'.format(kk))
            print('''Criteria: {0},
                Threshold: {1},
                If  Criteria > Threshold: {2},
                Segment: mean, median, max, and min:
                {3}, {4}, {5}, and {6}'''.format(cri, threshold, cri>=threshold, np.mean(cleaned)/factor, np.median(cleaned)/factor, np.max(cleaned)/factor, np.min(cleaned)/factor))
            if cri>=threshold:
                if whatT==1:
                    if j==2: mayZ3.append([np.mean(cleaned)/factor, how_many])
                    if j==1: mayY3.append([np.mean(cleaned)/factor, how_many])
                    if j==0: mayX3.append([np.mean(cleaned)/factor, how_many])
                if whatT==0:
                    if j==2: mayZ3.append([np.mean(cleaned)/factor, how_many])
                    if j==1: mayY3.append([np.mean(cleaned)/factor, how_many])
                    if j==0: mayX3.append([np.mean(cleaned)/factor, how_many])
                if whatT==2:
                    if j==2: mayZ3.append([np.mean(cleaned)/factor, how_many])
                    if j==1: mayY3.append([np.mean(cleaned)/factor, how_many])
                    if j==0: mayX3.append([np.mean(cleaned)/factor, how_many])
        pcdfiltered2[:,what]=pcdfiltered2[:,what]/factor
        points_in_each_seg.append(pcdfiltered2)
        print('---------- Plot ----------')
        plotter2(pcdfiltered2,"Rough Segmentation of the Structure toward {0} axis".format(names[j]),2,dfl,how_many, asr, (10,10))
    return mayX3, mayY3, mayZ3, points_in_each_seg, names

def refine_seg(mayX3, mayY3, mayZ3, whatT, names, pcd2, tj, cross1, asr):
    # Cluster similar segments toward each direction and refine the segmentations
    if whatT==1:
        print('---------- X segments ----------')
        mayX=ave_xy(np.array(mayX3)[:,0], 1)
        print('---------- Y segments ----------')
        mayY=ave_xy(np.array(mayY3)[:,0], 1)
        print('---------- Z segments ----------')
        mayZ=ave_xy(np.array(mayZ3)[:,0], 1)
    if whatT==0:
        print('---------- X segments ----------')
        mayX=ave_xy(np.array(mayX3)[:,0], 1)
        print('---------- Y segments ----------')
        mayY=ave_xy(np.array(mayY3)[:,0], 1)
        print('---------- Z segments ----------')
        mayZ=ave_xy(np.array(mayZ3)[:,0], 0.75)
    if whatT==2:
        print('---------- X segments ----------')
        mayX=ave_xy(np.array(mayX3)[:,0], 1)
        print('---------- Y segments ----------')
        mayY=ave_xy(np.array(mayY3)[:,0], 2)
        print('---------- Z segments ----------')
        mayZ=ave_xy(np.array(mayZ3)[:,0], 0.755)
    non_ceilings=np.asarray([[0, 0, 0], [0, 0, 0]])
    colored_points_list=np.asarray([[0, 0, 0, 0], [0, 0, 0, 0]])
    for jk in [0,1,2]:
        print('---------- Plot {0} segments ----------'.format(names[jk]))
        colored_points=segment_plotter(pcd2.copy(), jk, mayX , mayY, mayZ, tj,"Refined Segmentation of the Structure toward {0} axis".format(names[jk]), asr, (10,10))
        colored_points2=colored_points.copy()
        colored_points2[:,3]=jk
        colored_points_list=np.append(colored_points_list, colored_points2, axis=0)
        if jk!=2:
            non_ceilings=np.append(non_ceilings, colored_points[:, 0:3], axis=0)
    non_ceilings=non_ceilings[2:, :]
    non_ceilings=non_ceilings[non_ceilings[:, 2]>(pcd2[:, 2].min()+tj)]
    print(non_ceilings.shape)
    non_ceilings=np.append(non_ceilings, cross1, axis=0)

    return mayX, mayY, mayZ, colored_points_list,non_ceilings

def columns_seg(mayX, mayY, whatT, enormals, datao3d, asr):
    # This will consider the tilted ceiling
    SindY=np.asarray(mayY.copy())
    SindX=np.asarray(mayX.copy())
    SindY = list(SindY)
    SindY.sort()
    SindX = list(SindX)
    SindX.sort()
    # lets find the hieght at the location of each column.
    if whatT==2:
        diameter=2.5
    if whatT==1:
        diameter=1
    if whatT==0:
        diameter=5

    point_clouds,thexypoints=break_point_cloud(datao3d, SindX, SindY, enormals, diameter, 0.5)
    j=2
    what=j
    mayZ2=[]
    cte = np.linspace(0, 1, len(point_clouds))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for k in range(len(point_clouds)):
        mydata2=np.asarray(point_clouds[k].points)
        mayZ2.append(np.max(mydata2[:,what]))
        cte2=np.ones((mydata2.shape[0],1))*cte[k]
        print('Z = {0}  at  ( {1} , {2} )'.format(np.max(mydata2[:,what]), thexypoints[k][0], thexypoints[k][1]))
        # Optional plots
        scatter=ax.scatter(mydata2[:,0], mydata2[:,1], mydata2[:,2], c=cm.viridis(cte2), alpha=0.5, s=5)

    # Set labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_box_aspect(asr)

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, anchor=(1.2, 1.0))
    cbar.set_label('From (x_min, y_min) to (x_max, y_max)')

    print('---------- Plot ----------')
    # Set title
    ax.set_title("Max Z at each column")
    plt.show()
    return mayZ2

def node_gen(mayX, mayY, mayZ, pcdfiltered3, mayZ2):
    # Define the coordinate ranges

    x = list(np.array(mayX))
    x.sort()
    # x[0]= np.min(pcdfiltered3[:,0])
    y = list(np.array(mayY))
    y.sort()
    # y[0]= np.min(pcdfiltered3[:,1])

    z = list(np.array(mayZ))
    z.sort()

    # Add ground if it is not already considered in previous steps
    if len(z)>1 and z[0]<0.25:
        z[0]= np.min(pcdfiltered3[:,2])
    else:
        z=[np.min(pcdfiltered3[:,2])]+z


    # Generate raw nodes
    nodes=[]
    id = 1
    for k in z:
        for i in x:
            for j in y:
                nodes.append([i,j,k,id])
                id += 1

    # Correct nodes to consider a tilted ceiling
    nodes_arr=np.asarray(nodes)
    nodes2=nodes
    ind_ceil=np.where(nodes_arr[:,2]==z[-1])
    ind_ceil=np.sort(ind_ceil)
    nodes_arr[ind_ceil, 2]=np.asarray(mayZ2)
    nodes_L=nodes_arr.tolist()
    nodes=nodes_L

    print('''Ranges:
        x: {0}
        y: {1}
        z: {2}'''.format(x,y,z))


    xrang=np.array(x)
    yrang=np.array(y)
    zrang=np.array(z)
    if len(xrang)>1 and len(yrang)>1 and len(zrang)>1:
        difret=[xrang[1]-xrang[0], yrang[1]-yrang[0], zrang[1]-zrang[0]]
    else:
        if len(xrang)<=1:
            difret=[yrang[1]-yrang[0], zrang[1]-zrang[0]]
        if len(yrang)<=1:
            difret=[xrang[1]-xrang[0], zrang[1]-zrang[0]]
        if len(zrang)<=1:
            difret=[xrang[1]-xrang[0], yrang[1]-yrang[0]]

    Nodes_Errors=np.mean(1000*abs(np.asarray(nodes2)-np.asarray(nodes)), axis=0)[0:3]
    Nodes_Errors_max=np.max(1000*abs(np.asarray(nodes2)-np.asarray(nodes)), axis=0)[0:3]
    print('''X, Y, Z Errors (mm) between nodes:
        ''',Nodes_Errors)
    return nodes, nodes2, xrang, yrang, zrang, difret, Nodes_Errors, Nodes_Errors_max

def edge_gen(nodes2, nodes, mayX, mayY, mayZ2, asr):
    # Extract raw x, y, z, and id
    points2 = np.asarray(nodes2)

    # Generate raw straight edges
    edges2 = []

    for i in range(len(points2)):
        id1 = points2[i, 3]
        connections = [id1 + 1, id1+len(mayY), id1 + len(mayX)*len(mayY)]
        for id2 in connections:
            if id2 in points2[:, 3]:
                idx2 = find_index_by_id(points2, id2)
                x1, y1, z1, id1 = points2[i]
                x2, y2, z2, id2 = points2[idx2]
                dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
                edge2 = [
                    (x1, y1, z1, id1),
                    (x2, y2, z2, id2),
                    (x1 - x2, y1 - y2, z1 - z2, id1 - id2)
                ]
                if dx >= -0.5 and dy >= -0.5 and dz >= -0.5: edges2.append(edge2)


    # Extract corrected x, y, z, and id
    points = np.asarray(nodes)

    # Generate tilted edges
    edges = []

    for i in range(len(points)):
        id1 = points[i, 3]
        connections = [id1 + 1, id1+len(mayY), id1 + len(mayX)*len(mayY)]
        for id2 in connections:
                if id2 in points[:, 3]:
                    idx2 = find_index_by_id(points, id2)
                    x1, y1, z1, id1 = points[i]
                    x2, y2, z2, id2 = points[idx2]
                    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
                    edge = [
                        (x1, y1, z1, id1),
                        (x2, y2, z2, id2),
                        (x1 - x2, y1 - y2, z1 - z2, id1 - id2)
                    ]
                    if dx >= -0.5 and dy >= -0.5 and dz >= -0.5: edges.append(edge)

    aaaac=np.array(edges)
    # Plot edges
    fig = plt.figure(figsize=(10,20))

    ax = fig.add_subplot(311, projection='3d')

    for edge in edges:
        (x1, y1, z1, id1), (x2, y2, z2, id2), _ = edge
        ax.plot([x1, x2], [y1, y2], [z1, z2], marker='o')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title("Raw Wireframe with Corrected Ceiling Nodes")
    ax.set_box_aspect(asr)

    ax2 = fig.add_subplot(312, projection='3d')

    for edge2 in edges2:
        (x1, y1, z1, id1), (x2, y2, z2, id2), _ = edge2
        ax2.plot([x1, x2], [y1, y2], [z1, z2], marker='o')

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title("Raw Wireframe with Raw Ceiling Nodes")
    ax2.set_box_aspect(asr)

    ax3=fig.add_subplot(313)
    ax3.bar(np.asarray(nodes)[-len(mayZ2):,3], 1000*abs(np.asarray(nodes2)-np.asarray(nodes))[-len(mayZ2):, 2])
    ax3.set_xlabel('Node ID')
    ax3.set_ylabel('Correction (mm)')
    ax3.set_title("Corrected vs Raw Ceiling Nodes")

    plt.show()

    fig = plt.figure(figsize=(10,5))

    # 2D plot for x,z plane
    ax_xz = fig.add_subplot(121)
    for edge in edges:
        (x1, y1, z1, id1), (x2, y2, z2, id2), _ = edge
        ax_xz.plot([x1, x2], [z1, z2], marker='o')
    ax_xz.set_xlabel('X (m)')
    ax_xz.set_ylabel('Z (m)')
    ax_xz.set_title("X-Z Plane")

    # 2D plot for y,z plane
    ax_yz = fig.add_subplot(122)
    for edge in edges:
        (x1, y1, z1, id1), (x2, y2, z2, id2), _ = edge
        ax_yz.plot([y1, y2], [z1, z2], marker='o')
    ax_yz.set_xlabel('Y (m)')
    ax_yz.set_ylabel('Z (m)')
    ax_yz.set_title("Y-Z Plane")

    plt.tight_layout()
    plt.show()

    return aaaac

def vis_edges(aaaac, pcdtocross, nodes, difret, asr, W, depth=0.5):
    # Visualize tilted edges
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')


    for edge in aaaac:
        x = [edge[0, 0], edge[1, 0]]
        y = [edge[0, 1], edge[1, 1]]
        z = [edge[0, 2], edge[1, 2]]
        ax.plot(x, y, z, 'b-',linewidth=2)

    x = [edge[0, 0], edge[1, 0]]
    y = [edge[0, 1], edge[1, 1]]
    z = [edge[0, 2], edge[1, 2]]
    ax.plot(x, y, z, 'b-', label="Corrected Elements",linewidth=0.5)

    for edge in aaaac:
        x = [edge[0, 0] , edge[1, 0]]
        y = [edge[0, 1] , edge[1, 1]]
        z = [edge[0, 2] , edge[1, 2]]
        ax.scatter(x, y, z, 'b-', c="red", s=50)

    x = [edge[0, 0] , edge[1, 0]]
    y = [edge[0, 1] , edge[1, 1]]
    z = [edge[0, 2] , edge[1, 2]]
    ax.scatter(x, y, z, 'b-', c="red", label="Node", s=50)

    # Compute mid points
    midss=[]
    for edge in aaaac:
        x = (edge[0, 0] + edge[1, 0])/2
        y = (edge[0, 1] + edge[1, 1])/2
        z = (edge[0, 2] + edge[1, 2])/2
        midss.append([x,y,z])

    # Set labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend(bbox_to_anchor=(1.2, 1.0), loc='upper left')
    ax.set_box_aspect(asr)
    # Set title
    ax.set_title("Nodes and Elements - Corrected Elements")

    # Plot the PCD
    fig = plt.figure(figsize=(10, 10))
    ax2 = fig.add_subplot(111, projection='3d')

    if W==2:
        pcdtocross=pcdtocross[pcdtocross[:, 2]<pcdtocross[:, 2].max()-depth]
        ax2.scatter(pcdtocross[:,0], pcdtocross[:,1], pcdtocross[:,2], c="green", alpha=0.1, s=5, label="Point Cloud without Ground")
    if W==0 or W==1:
        ax2.scatter(pcdtocross[:,0], pcdtocross[:,1], pcdtocross[:,2], c="green", alpha=0.1, s=5, label="Point Cloud without Ground")

    # Set labels
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')

    # Set title
    ax2.set_title("Point Cloud without Ground")
    ax2.legend(bbox_to_anchor=(1.2, 1.0), loc='upper left')
    ax2.set_box_aspect(asr)
    # Generate additional tilted edges
    nedges2, coord_diff, midss22 = find_edges(nodes, difret, 2)
    aaaac2=np.array(coord_diff)

    # Plot additional tilted edges
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for edge in aaaac2:
        x = [edge[0, 0], edge[1, 0]]
        y = [edge[0, 1], edge[1, 1]]
        z = [edge[0, 2], edge[1, 2]]
        ax.plot(x, y, z, 'b-',linewidth=0.5)

    x = [edge[0, 0], edge[1, 0]]
    y = [edge[0, 1], edge[1, 1]]
    z = [edge[0, 2], edge[1, 2]]
    ax.plot(x, y, z, 'b-', label="Additional Tilted Elements",linewidth=0.5)

    for edge in aaaac2:
        x = [edge[0, 0] , edge[1, 0]]
        y = [edge[0, 1] , edge[1, 1]]
        z = [edge[0, 2] , edge[1, 2]]
        ax.scatter(x, y, z, 'b-', c="red", s=50)

    x = [edge[0, 0] , edge[1, 0]]
    y = [edge[0, 1] , edge[1, 1]]
    z = [edge[0, 2] , edge[1, 2]]
    ax.scatter(x, y, z, 'b-', c="red", label="Node", s=50)

    # Set labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend(bbox_to_anchor=(1.2, 1.0), loc='upper left')

    # Set title
    ax.set_title("Nodes and Elements - Additional Tilted Elements")
    ax.set_box_aspect(asr)
    plt.show()
    return nedges2, coord_diff, midss22, aaaac2, midss


import numpy as np

def slice_array_by_z(array, dz=0.5):
    # Sort the array by the z column
    sorted_array = array[array[:, 2].argsort()]
    
    # Create a list to store the slices
    slices = []
    
    # Initialize the start index
    start_idx = 0
    
    # Loop through the array and slice it based on dz
    while start_idx < len(sorted_array):
        # Find the end index for the current slice
        end_idx = start_idx
        while end_idx < len(sorted_array) and sorted_array[end_idx, 2] < sorted_array[start_idx, 2] + dz:
            end_idx += 1
        
        # Append the current slice to the list
        slices.append(sorted_array[start_idx:end_idx])
        
        # Update the start index for the next slice
        start_idx = end_idx
    
    return slices

def add_wall_seg(whatT, pcd2, my_add, wall_th, spli, bin, walls0, asr):
    walls=np.array([[0,0,0],[0,0,0]])
    if whatT==2:
        my_perc=95
        my_width=0.5
    if whatT==1:
        my_perc=95
        my_width=1
    if whatT==0:
        my_perc=99
        my_width=0.5
    if len(my_add[1]):
        for data in my_add[1]:
            tiba=0
            sl_pcd=slice_array_by_z(data, spli)
            for hist_data in sl_pcd:
                aaaaaaaaa=compute_and_plot_histogram_with_peaks(my_perc, hist_data, tiba, bin, my_width, 0)
                counter=0
                for ranger in aaaaaaaaa[5]:
                    R_min=np.asarray(ranger)[0]
                    R_max=np.asarray(ranger)[1]
                    plane=points_within_range(hist_data, R_min, R_max, tiba)
                    print('-------------- X planes --------------')
                    print('''The Hist of the wall candidate is: {0}
                        The Hist threshold to be a wall is: {1}
                        The wall pass: {2}'''.format(aaaaaaaaa[8][counter], wall_th, aaaaaaaaa[8][counter]>wall_th))
                    if aaaaaaaaa[8][counter]>wall_th:
                        walls=np.append(walls, plane, axis=0)
                    counter +=1
    if len(my_add[2]):
        for data in my_add[2]:
            tiba=1
            sl_pcd=slice_array_by_z(data, spli)
            for hist_data in sl_pcd:
                aaaaaaaaa=compute_and_plot_histogram_with_peaks(my_perc, hist_data, tiba, bin, my_width, 0)
                counter=0
                for ranger in aaaaaaaaa[5]:
                    R_min=np.asarray(ranger)[0]
                    R_max=np.asarray(ranger)[1]
                    plane=points_within_range(hist_data, R_min, R_max, tiba)
                    print('-------------- Y planes --------------')
                    print('''The Hist of the wall candidate is: {0}
                        The Hist threshold to be a wall is: {1}
                        The wall pass: {2}'''.format(aaaaaaaaa[8][counter], wall_th, aaaaaaaaa[8][counter]>wall_th))
                    if aaaaaaaaa[8][counter]>wall_th:
                        walls=np.append(walls, plane, axis=0)
                    counter +=1

    walls=walls[2:, :]
    # Find the rows in pcd2 that are not in walls
    wall_mask = np.all(np.isin(pcd2, walls), axis=1)

    if len(walls0) or len(walls):
        walls=np.append(walls0, walls, axis=0)
    plotter2(walls, 'Walls', 0, 0, 0, asr, size=(10, 10))
    return walls, wall_mask

def wall_ceiling_seg(whatT, pcd2):
    walls=np.array([[0,0,0],[0,0,0]])
    Ceiling2=np.array([[0,0,0],[0,0,0]])
    hist_data=pcd2.copy()
    # wall_th=300
    wall_th=0.04
    myz=[]
    myx_wall=[]
    myy_wall=[]
    myx_element=[]
    myy_element=[]
    myz_element=[]
    if whatT==2:
        my_perc=95
        my_width=0.5
    if whatT==1:
        my_perc=94
        my_width=1
    if whatT==0:
        my_perc=90
        my_width=0.5
    for tiba in [0,1,2]:
        aaaaaaaaa=compute_and_plot_histogram_with_peaks(my_perc, hist_data, tiba, 0.15, my_width, 1)
        if tiba==0:
            myx_element.append([aaaaaaaaa[4], aaaaaaaaa[8]])
        if tiba==1: 
            myy_element.append([aaaaaaaaa[4], aaaaaaaaa[8]])
        if tiba==2: 
            myz_element.append([aaaaaaaaa[4], aaaaaaaaa[8]])
        counter=0
        for ranger in aaaaaaaaa[5]:
            R_min=np.asarray(ranger)[0]
            R_max=np.asarray(ranger)[1]
            plane=points_within_range(hist_data, R_min, R_max, tiba)
            if tiba==0:
                print('-------------- X planes --------------')
                print('''The Hist of the wall candidate is: {0}
                    The Hist threshold to be a wall is: {1}
                    The wall pass: {2}'''.format(aaaaaaaaa[8][counter], wall_th, aaaaaaaaa[8][counter]>wall_th))
                if aaaaaaaaa[8][counter]>wall_th:
                    walls=np.append(walls, plane, axis=0)
                myx_wall.append(plane)
                counter +=1

            if tiba==1:
                print('-------------- Y planes --------------')
                print('''The Hist of the wall candidate is: {0}
                    The Hist threshold to be a wall is: {1}
                    The wall pass: {2}'''.format(aaaaaaaaa[8][counter], wall_th, aaaaaaaaa[8][counter]>wall_th))
                if aaaaaaaaa[8][counter]>wall_th:
                    walls=np.append(walls, plane, axis=0)
                myy_wall.append(plane)
                counter +=1

            if tiba==2:
                Ceiling2=np.append(Ceiling2, plane, axis=0)
                myz.append(plane[:, 2].mean())

    walls=walls[2:, :]
    # Find the rows in pcd2 that are not in walls
    wall_mask = np.all(np.isin(pcd2, walls), axis=1)

    Ceiling2=Ceiling2[2:, :]
    # Find the rows in pcd2 that are not in walls
    ceiling_mask = np.all(np.isin(pcd2, Ceiling2), axis=1)

    return walls, Ceiling2, [wall_mask, ceiling_mask], [myz, myx_wall, myy_wall], [myx_element, myy_element, myz_element]

def find_shared_indices_above_percentile(arr, col1, col2, percentile):
    if percentile<100:
        # Compute the percentile values for each column
        percentile_value_col1 = np.percentile(arr[:, col1], percentile)
        percentile_value_col2 = np.percentile(arr[:, col2], percentile)
        
        # Find indices of elements in each column that are higher than the percentile value
        indices_col1 = np.where(arr[:, col1] >= percentile_value_col1)[0]
        indices_col2 = np.where(arr[:, col2] >= percentile_value_col2)[0]
        
        # Find the shared indices between the two columns
        shared_indices = np.intersect1d(indices_col1, indices_col2)
        if len(shared_indices)>0:
            shared_indices = np.intersect1d(indices_col1, indices_col2)
        if len(indices_col2)>0:
            shared_indices=indices_col2
        if len(indices_col1)>0:
            shared_indices=indices_col1

        # Find them
        arr_valid=arr[shared_indices, :]

        # Find min area
        area_array=arr_valid[:, col1]*arr_valid[:, col1]
        min_area_ind=np.where(area_array==area_array.min())[0]

        # Find values with min area
        arr_valid_out=arr_valid[min_area_ind, :]
    else:
        # Compute the percentile values for each column
        percentile_value_col1 = np.percentile(arr[:, col1], percentile-100)
        percentile_value_col2 = np.percentile(arr[:, col2], percentile-100)
        percentile_value_col12 = np.percentile(arr[:, col1], 100-(percentile-100))
        percentile_value_col22 = np.percentile(arr[:, col2], 100-(percentile-100))        

        # Find indices of elements in each column that are higher than the percentile value
        indices_col1 = np.where((arr[:, col1] <= percentile_value_col12)) 
        indices_col2 = np.where((arr[:, col2] <= percentile_value_col22))
        
        # Find the shared indices between the two columns
        shared_indices = np.intersect1d(indices_col1, indices_col2)
        if len(shared_indices)>0:
            shared_indices = np.intersect1d(indices_col1, indices_col2)
        if len(indices_col2)>0:
            shared_indices=indices_col2
        if len(indices_col1)>0:
            shared_indices=indices_col1

        # Find them
        arr_valid=arr[shared_indices, :]

        # Find min area
        area_array=arr_valid[:, col1]*arr_valid[:, col1]
        min_area_ind=np.where(area_array==area_array.min())[0]

        # Find values with min area
        arr_valid_out=np.array([np.mean((arr_valid[min_area_ind, :])[0], axis=0)])
    return arr_valid_out

import numpy as np

def find_nodes_under_conditions(WandL, start, end, threshold, id):
    nodes3 = np.vstack((np.array(WandL[1:])[:, 3:6], np.array(WandL[1:])[:, 6:9]))
    
    x1, y1, z1 = start
    x2, y2, z2 = end
    
    if z1 < z2:
        x_ref, y_ref, z_ref = x1, y1, z1
    else:
        x_ref, y_ref, z_ref = x2, y2, z2

    result_nodes = []
    if id==1 and z_ref > nodes3[:,2].min():
        for node in nodes3:
            x3, y3, z3 = node
            if (z_ref-z3) > threshold  and np.isclose(x3, x_ref, atol=threshold) and np.isclose(y3, y_ref, atol=threshold):
                result_nodes.append(node)
    else:
        result_nodes.append([x_ref, y_ref, z_ref])
    
    return np.array(result_nodes)

import numpy as np
from collections import defaultdict

def fix_the_graph(WandL, new_WandL_ind, whatT):
    if whatT==2:
        # Example numpy array of lines: each row is a line with start and end points
        lines = (np.array((np.array(WandL)[new_WandL_ind]).tolist())).copy()

        # Dictionary to count occurrences of each point
        point_count = defaultdict(int)

        # Count each point
        for line in lines:
            start = tuple(line[3:6])
            end = tuple(line[6:9])
            point_count[start] += 1
            point_count[end] += 1

        # Find points that are shared between only two lines
        shared_points = [point for point, count in point_count.items() if count == 2]

        # Find the indices of lines that contain the shared points
        line_indices = []
        line_indices_xys1 = []
        line_indices_xys2 = []
        line_indices_xys3 = []
        for i, line in enumerate(lines):
            start = tuple(line[3:6])
            end = tuple(line[6:9])
            if start in shared_points or end in shared_points:
                line_indices.append(i)
                if start in shared_points and end in shared_points:
                    line_indices_xys1.append(line.tolist() + [i])
                elif start in shared_points:
                    line_indices_xys2.append(line.tolist() + [i])
                elif end in shared_points:
                    line_indices_xys3.append(line.tolist() + [i])

        new_lines = []
        lines_to_remove = set()

        # First loop

        not_rot_points=[]
        deep_curve=[]
        for i, line in enumerate(line_indices_xys1):
            start_shared = line[3:6]
            end_shared = line[6:9]
            not_rot_points.append(start_shared)
            not_rot_points.append(end_shared)

            a1 = [k for k, v in enumerate(line_indices_xys3) if tuple(v[6:9]) == tuple(start_shared)]
            a2 = [k for k, v in enumerate(line_indices_xys2) if tuple(v[3:6]) == tuple(end_shared)]

            if a1 and a2:
                a1_index = a1[-1]
                a2_index = a2[-1]

                lines_to_remove.update([line_indices_xys3[a1_index][-1], line_indices_xys2[a2_index][-1], line[-1]])
                
                line_indices_xys3[a1_index][15]=404
                line_indices_xys2[a2_index][15]=404
                line[15]=404
                new_lines.append(np.array(line_indices_xys3[a1_index][0:-1]))
                new_lines.append(np.array(line_indices_xys2[a2_index][0:-1]))
                new_lines.append(np.array(line[0:-1]))
                deep_curve.append([np.array(line_indices_xys3[a1_index][0:-1]), np.array(line_indices_xys2[a2_index][0:-1]), np.array(line[0:-1])])


        # Second loop
        for j, line in enumerate(line_indices_xys2):
            a3 = [k for k, v in enumerate(line_indices_xys3) if tuple(v[6:9]) == tuple(line[3:6])]

            if a3:
                a3_index = a3[-1]

                lines_to_remove.update([line_indices_xys3[a3_index][-1], line[-1]])

                properties = ((np.array(line_indices_xys3[a3_index]) + np.array(line)) / 2).tolist()
                new_start = line_indices_xys3[a3_index][3:6]
                new_end = line[6:9]
                new_diff= (np.array(new_start) - np.array(new_end)).tolist()
                properties[19]=np.sqrt(((new_start[0]-new_end[0])**2+(new_start[1]-new_end[1])**2+(new_start[2]-new_end[2])**2))

                new_lines.append(np.array(properties[0:3] + new_start + new_end + new_diff + properties[12:-1]))

        # Remove the old lines
        lines = np.delete(lines, list(lines_to_remove), axis=0)

        # Add the new merged lines
        lines = np.vstack((lines, new_lines))

        return lines.tolist(), not_rot_points, deep_curve
    else:
        return WandL, [], []

def refine_nodes_elements_refinement(WandL, whatT, vis_it):

    if whatT==2:
        list_checker=[]
        for i_check in range(len(WandL)):
            if WandL[i_check][1]>=WandL[i_check][0]:
                bbbb=WandL[i_check][1]
                cccc=WandL[i_check][0]
            else:
                bbbb=WandL[i_check][0]
                cccc=WandL[i_check][1]
            if bbbb!=0:
                aaaa=(cccc/bbbb)
            else:
                aaaa=0
            if aaaa>0.16:
                acheck=find_nodes_under_conditions(WandL, WandL[i_check][3:6], WandL[i_check][6:9], 0.05, WandL[i_check][15])
                if len(acheck):
                    # print(i_check)
                    list_checker.append([i_check, np.round(aaaa, 3), (WandL[i_check][0]), (WandL[i_check][1])])
        np_ind_s_e=np.array(list_checker, dtype=int)
        if vis_it:
            # Visualize the edges
            fig = plt.figure(figsize=(10, 10))
            axg = fig.add_subplot(111, projection='3d')



            avvr=np.array(WandL)

            for edge in avvr:
                if not (edge[0]==0 and edge[1]==0):
                    x = [edge[3], edge[6]]
                    y = [edge[4], edge[7]]
                    z = [edge[5], edge[8]]
                    axg.plot(x, y, z, 'b-',linewidth=2)

            x = [edge[3], edge[6]]
            y = [edge[4], edge[7]]
            z = [edge[5], edge[8]]
            axg.plot(x, y, z, 'b-', label="Refined Element",linewidth=2)

            for edge in avvr:
                if not (edge[0]==0 and edge[1]==0):
                    x = [edge[3], edge[6]]
                    y = [edge[4], edge[7]]
                    z = [edge[5], edge[8]]
                    axg.scatter(x, y, z, 'b-', c="red", s=50)

            x = [edge[3], edge[6]]
            y = [edge[4], edge[7]]
            z = [edge[5], edge[8]]
            axg.scatter(x, y, z, 'b-', c="red", label="Refined Node", s=50)

            avvr=np.array(WandL)[np_ind_s_e[:, 0]]
            text=np.array(list_checker, dtype=str)[:, 1]

            itext=0
            for edge in avvr:
                if not (edge[0] == 0 and edge[1] == 0):
                    x = [edge[3], edge[6]]
                    y = [edge[4], edge[7]]
                    z = [edge[5], edge[8]]
                    axg.plot(x, y, z, 'r-', linewidth=5)
                    mid_x = (x[0] + x[1]) / 2
                    mid_y = (y[0] + y[1]) / 2
                    mid_z = (z[0] + z[1]) / 2
                    axg.text(mid_x, mid_y, mid_z, text[itext], size=12, color='black')
                    itext +=1

            x = [edge[3], edge[6]]
            y = [edge[4], edge[7]]
            z = [edge[5], edge[8]]
            axg.plot(x, y, z, 'r-', label="Remained Element",linewidth=5)

            for edge in avvr:
                if not (edge[0]==0 and edge[1]==0):
                    x = [edge[3], edge[6]]
                    y = [edge[4], edge[7]]
                    z = [edge[5], edge[8]]
                    axg.scatter(x, y, z, 'b-', c="black", s=100)

            x = [edge[3], edge[6]]
            y = [edge[4], edge[7]]
            z = [edge[5], edge[8]]
            axg.scatter(x, y, z, 'b-', c="black", label="Node", s=100)

            # Set labels
            axg.set_xlabel('X (m)')
            axg.set_ylabel('Y (m)')
            axg.set_zlabel('Z (m)')

            axg.set_box_aspect(asr)
            axg.legend(bbox_to_anchor=(1.2, 1.0), loc='upper left')

            # Set title
            axg.set_title("Refined Nodes and Elements")

        return np_ind_s_e[:, 0]
    else:
        return np.ones(len(WandL), dtype=int).tolist()



def compute_nodes_elements_refinement(point, hjkl, point_tree2, diff, Counter_check, what_shape, plotkon):
    properties = []
    if point_tree2.query_ball_point(point, hjkl):
        around_each = point_tree2.data[point_tree2.query_ball_point(point, hjkl)]
        results = [project_point_onto_plane(p, point, np.array([diff[0], diff[1], diff[2]])) for p in around_each]
        around_each_proj = np.array([res[1] for res in results])
        conversion_scales = np.array([res[2] for res in results])
        # Use the average conversion scale for simplicity
        average_conversion_scale = np.mean(conversion_scales)
        our_prop = compute_area_and_moments(around_each_proj, Counter_check, what_shape, plotkon, average_conversion_scale)
        properties.append([our_prop[0], our_prop[1], our_prop[2], our_prop[3], our_prop[4], our_prop[5]])
    return properties

def process_chunk0(chunk, hjkl, point_tree2, diff, Counter_check, what_shape, plotkon):
    properties = []
    for point in chunk:
        properties.extend(compute_nodes_elements_refinement(point, hjkl, point_tree2, diff, Counter_check, what_shape, plotkon))
    return properties

def parallel_nodes_elements_refinement(discretized_points2, hjkl, point_tree2, diff, Counter_check, what_shape, plotkon, chunk_size):
    chunks = [discretized_points2[i:i + chunk_size] for i in range(0, len(discretized_points2), chunk_size)]
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_chunk0, chunk, hjkl, point_tree2, diff, Counter_check, what_shape, plotkon) for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
    return results

def project_parallel(p, mids, diff):
    return project_point_onto_plane(p, mids, diff)[0]

def compute_loop_of_nodes_elements_refinement(aaaac, i, point_tree, point_tree2, walls, hjkl, perc, WandL, Arounds, what_shape, plotkon, Counter_check, Counter_check_beam, Counter_check_column, all, proj, midss, num_points, PWandL, WandLN, modddd):
    if modddd==2:
        ped1xyz=aaaac[i,0,0:4]
        ped1xyz.reshape(1,4)
        ped2xyz=aaaac[i,1,0:4]
        ped2xyz.reshape(1,4)
        diff=aaaac[i,2,0:3]
        mids=np.array(midss)[i]
        PWandL.append([mids[0],mids[1],mids[2], ped1xyz[0], ped1xyz[1], ped1xyz[2], ped2xyz[0], ped2xyz[1], ped2xyz[2], diff[0], diff[1], diff[2]])
        around= point_tree.data[point_tree.query_ball_point([mids[0],mids[1],mids[2]], hjkl)]
        discretized_points = discretize_line(ped1xyz[0:3], ped2xyz[0:3], num_points)
        all_points_around = True
        for point in discretized_points:
            if not point_tree.query_ball_point(point, hjkl):
                all_points_around = False
                break
        around=np.array(around)
        if not all_points_around:
            WandLN.append([0,0,i, ped1xyz[0], ped1xyz[1], ped1xyz[2], ped2xyz[0], ped2xyz[1], ped2xyz[2], diff[0], diff[1], diff[2], 0, 0, 0, 0, ped1xyz[3], ped2xyz[3]])
        if (all_points_around and around.shape[0]):
            Counter_check=Counter_check+1
            if walls.shape[0]:
                # Compare walls with the columns of around
                # Extract the first three columns of around
                pcd_first_three = around.copy()
                # Find the rows in around that are in walls
                maskW = np.all(np.isin(pcd_first_three, walls), axis=1)
            else:
                maskW=[]
            len_of_element=np.sqrt(diff[0]**2+diff[1]**2+diff[2]**2)
            sin_ang_of_element=np.sqrt(diff[0]**2+diff[1]**2)/len_of_element
            if sin_ang_of_element > 0.34: # 0.34 is Sin(20 degree)
                Counter_check_beam += 1
                if np.any(maskW):
                    Eid=300
                if not np.any(maskW):
                  if abs(diff[0]) > abs(diff[1]):
                      Eid=100
                  if abs(diff[0]) < abs(diff[1]):
                      Eid=10
                around2=np.hstack((around,np.ones([around.shape[0],1])*Eid))
                all=np.append(around2, all, axis=0)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    around_proj_list = list(executor.map(lambda p: project_parallel(p, np.array([mids[0], mids[1], mids[2]]), np.array([diff[0], diff[1], diff[2]])), around))
                around_proj = np.array(around_proj_list)
                properties=[]
                discretized_points2 = discretize_line(ped1xyz[0:3], ped2xyz[0:3], 10)
                properties=parallel_nodes_elements_refinement(discretized_points2, hjkl, point_tree2, diff, Counter_check, what_shape, plotkon, chunk_size=int(np.floor(my_chunk*len(discretized_points2))))
                properties = np.array(properties)
                properties=remove_zeros_and_nan(properties)
                properties_val = find_shared_indices_above_percentile(properties, 4, 5, perc)
                AreA, Ix, Iy, Jx_y, length, width = properties_val[0]
                WandL.append([width,length,i, ped1xyz[0], ped1xyz[1], ped1xyz[2], ped2xyz[0], ped2xyz[1], ped2xyz[2], diff[0], diff[1], diff[2], Ix, Iy, Jx_y, Eid, ped1xyz[3], ped2xyz[3], AreA, len_of_element])
                around2=np.hstack((around_proj,np.ones([around.shape[0],1])*Eid))
                proj=np.append(around2, proj, axis=0)
                Arounds.append([all, proj, i])
            elif sin_ang_of_element < 0.34: # 0.34 is Sin(20 degree)
                Counter_check_column +=1
                if np.any(maskW):
                    Eid=300
                else:
                    Eid=1
                around2=np.hstack((around,np.ones([around.shape[0],1])*Eid))
                all=np.append(around2, all, axis=0)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    around_proj_list = list(executor.map(lambda p: project_parallel(p, np.array([mids[0], mids[1], mids[2]]), np.array([diff[0], diff[1], diff[2]])), around))
                around_proj = np.array(around_proj_list)                
                properties=[]
                discretized_points2 = discretize_line(ped1xyz[0:3], ped2xyz[0:3], 10)
                properties=parallel_nodes_elements_refinement(discretized_points2, hjkl, point_tree2, diff, Counter_check, what_shape, plotkon, chunk_size=int(np.floor(my_chunk*len(discretized_points2))))
                properties = np.array(properties)
                properties=remove_zeros_and_nan(properties)
                properties_val = find_shared_indices_above_percentile(properties, 4, 5, perc)
                AreA, Ix, Iy, Jx_y, length, width = properties_val[0]
                WandL.append([width,length,i, ped1xyz[0], ped1xyz[1], ped1xyz[2], ped2xyz[0], ped2xyz[1], ped2xyz[2], diff[0], diff[1], diff[2], Ix, Iy, Jx_y, Eid, ped1xyz[3], ped2xyz[3], AreA, len_of_element])
                around2=np.hstack((around_proj,np.ones([around.shape[0],1])*Eid))
                proj=np.append(around2, proj, axis=0)
                Arounds.append([all, proj, i])
    if modddd==1:
        ped1xyz=aaaac[i,0,0:4]
        ped1xyz.reshape(1,4)
        ped2xyz=aaaac[i,1,0:4]
        ped2xyz.reshape(1,4)
        diff=aaaac[i,2,0:3]
        mids=np.array(midss)[i,0,:]
        PWandL.append([mids[0],mids[1],mids[2], ped1xyz[0], ped1xyz[1], ped1xyz[2], ped2xyz[0], ped2xyz[1], ped2xyz[2], diff[0], diff[1], diff[2]])
        around= point_tree.data[point_tree.query_ball_point([mids[0],mids[1],mids[2]], hjkl)]
        discretized_points = discretize_line(ped1xyz[0:3], ped2xyz[0:3], num_points)
        all_points_around = True
        for point in discretized_points:
            if not point_tree.query_ball_point(point, hjkl):
                all_points_around = False
                break
        around=np.array(around)
        if not all_points_around:
            WandLN.append([0,0,i, ped1xyz[0], ped1xyz[1], ped1xyz[2], ped2xyz[0], ped2xyz[1], ped2xyz[2], diff[0], diff[1], diff[2], 0, 0, 0, 0, ped1xyz[3], ped2xyz[3]])
        if (all_points_around and around.shape[0]):
            Counter_check=Counter_check+1
            if walls.shape[0]:
                # Compare walls with the columns of around
                # Extract the first three columns of around
                pcd_first_three = around.copy()
                # Find the rows in around that are in walls
                maskW = np.all(np.isin(pcd_first_three, walls), axis=1)
            else:
                maskW=[]
            len_of_element=np.sqrt(diff[0]**2+diff[1]**2+diff[2]**2)
            if np.any(maskW):
                Eid=300
            else:
                Eid=200
            around2=np.hstack((around,np.ones([around.shape[0],1])*Eid))
            all=np.append(around2, all, axis=0)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                around_proj_list = list(executor.map(lambda p: project_parallel(p, np.array([mids[0], mids[1], mids[2]]), np.array([diff[0], diff[1], diff[2]])), around))
            around_proj = np.array(around_proj_list)
            properties=[]
            discretized_points2 = discretize_line(ped1xyz[0:3], ped2xyz[0:3], 10)
            properties=parallel_nodes_elements_refinement(discretized_points2, hjkl, point_tree2, diff, Counter_check, what_shape, plotkon, chunk_size=int(np.floor(my_chunk*len(discretized_points2))))
            properties = np.array(properties)
            properties=remove_zeros_and_nan(properties)
            properties_val = find_shared_indices_above_percentile(properties, 4, 5, perc)
            AreA, Ix, Iy, Jx_y, length, width = properties_val[0]
            WandL.append([width,length,i, ped1xyz[0], ped1xyz[1], ped1xyz[2], ped2xyz[0], ped2xyz[1], ped2xyz[2], diff[0], diff[1], diff[2], Ix, Iy, Jx_y, Eid, ped1xyz[3], ped2xyz[3], AreA, len_of_element])
            around2=np.hstack((around_proj , np.ones([around_proj .shape[0],1])*Eid))
            proj=np.append(around2, proj, axis=0)
            Arounds.append([all, proj, i])
    return Counter_check, Counter_check_beam, Counter_check_column, WandL, proj, all, WandLN, PWandL, Arounds

my_chunk=1

def nodes_elements_refinement(Nodes_Errors, pcd2, whatT, aaaac2, aaaac, walls, midss22, midss, pcdtocross, plotkon, what_shape, perc, depth=0.5):
    # Number of points to discretize the line into
    num_points = 1000
    # Simple assumption of rectangle
    # or set it to poly for polygon fitting
    # or set it to other for other fitting
    if whatT==1:
        hjkl=1
        if not perc:
            perc=50
    if whatT==0:
        hjkl=0.6
        if not perc:
            perc=5
    if whatT==2:
        hjkl=0.5
        pcdtocross=pcdtocross[pcdtocross[:, 2]<pcdtocross[:, 2].max()-depth]
        if not perc:
            perc=10

    point_tree = spatial.cKDTree(pcdtocross.copy())
    point_tree2 = spatial.cKDTree(pcd2.copy())

    WandLN=[]
    WandL=[]
    Arounds=[]
    PWandL=[]
    Counter_check=0
    Counter_check_beam=0
    Counter_check_column=0
    all=np.array([[0,0,0,0],[0,0,0,0]])
    proj=np.array([[0,0,0,0],[0,0,0,0]])
    WandL.append([0] * 20)

    # Additional Tilted - Cross-section extraction
    process=1
    process_100=len(aaaac2)
    for i in range(aaaac2.shape[0]):
        Counter_check, Counter_check_beam, Counter_check_column, WandL, proj, all, WandLN, PWandL, Arounds=compute_loop_of_nodes_elements_refinement(aaaac2, i, point_tree, point_tree2, walls, hjkl, perc, WandL, Arounds, what_shape, plotkon, Counter_check, Counter_check_beam, Counter_check_column, all, proj, midss22, num_points, PWandL, WandLN, 1)
        print('{0}'.format(np.ceil(100*(process/process_100))), 'percent is done', end="\r")
        process +=1

    print("Correct Additional Tilted Elements Number= ", Counter_check)


    # Corrected Tilted - Cross-section extraction
    num_points=5
    Counter_check=0
    process=1
    process_100=len(aaaac)
    for i in range(aaaac.shape[0]):
        Counter_check, Counter_check_beam, Counter_check_column, WandL, proj, all, WandLN, PWandL, Arounds= compute_loop_of_nodes_elements_refinement(aaaac, i, point_tree, point_tree2, walls, hjkl, perc, WandL, Arounds, what_shape, plotkon, Counter_check, Counter_check_beam, Counter_check_column, all, proj, midss, num_points, PWandL, WandLN, 2)
        print('{0}'.format(np.ceil(100*(process/process_100))), 'percent is done', end="\r")
        process +=1
    # parallel_process(mode=2)
    print("Correct Tilted Elements Number= ", Counter_check)
    print("Correct Tilted Beam Number= ", Counter_check_beam)
    print("Correct Tilted Column Number= ", Counter_check_column)
    return WandL, proj[2:], all[2:], WandLN, PWandL, Arounds

import numpy as np
import concurrent.futures
from scipy import spatial

def compute_geo_p_of_geo_prop_refinement(point, hjkl, point_tree2, rad_tresh, diff, Counter_check, what_shape, plotkon):
    properties = []
    if point_tree2.query_ball_point(point, hjkl):
        if rad_tresh != hjkl:
            radius = 0.00001
            around_each = []
            while not len(around_each):
                prev_radius = radius
                radius += 0.075
                current_points = point_tree2.data[point_tree2.query_ball_point(point, radius)]
                new_points = [p for p in current_points if np.linalg.norm(p - point) > prev_radius]
                around_each.extend(new_points)
            while not (len(new_points) == 0 or radius >= rad_tresh):
                prev_radius = radius
                radius += 0.075
                current_points = point_tree2.data[point_tree2.query_ball_point(point, radius)]
                new_points = [p for p in current_points if np.linalg.norm(p - point) > prev_radius]
                around_each.extend(new_points)
            if len(around_each) > 0:
                results = [project_point_onto_plane(p, point, np.array([diff[0], diff[1], diff[2]])) for p in around_each]
                around_each_proj = np.array([res[1] for res in results])
                conversion_scales = np.array([res[2] for res in results])
                average_conversion_scale = np.mean(conversion_scales)
                our_prop = compute_area_and_moments(around_each_proj, Counter_check, what_shape, plotkon, average_conversion_scale)
                properties.append([our_prop[0], our_prop[1], our_prop[2], our_prop[3], our_prop[4], our_prop[5]])
        else:
            around_each = point_tree2.data[point_tree2.query_ball_point(point, hjkl)]
            results = [project_point_onto_plane(p, point, np.array([diff[0], diff[1], diff[2]])) for p in around_each]
            around_each_proj = np.array([res[1] for res in results])
            conversion_scales = np.array([res[2] for res in results])
            average_conversion_scale = np.mean(conversion_scales)
            our_prop = compute_area_and_moments(around_each_proj, Counter_check, what_shape, plotkon, average_conversion_scale)
            properties.append([our_prop[0], our_prop[1], our_prop[2], our_prop[3], our_prop[4], our_prop[5]])
    return properties

def process_chunk(chunk, hjkl, point_tree2, rad_tresh, diff, Counter_check, what_shape, plotkon):
    properties = []
    for point in chunk:
        properties.extend(compute_geo_p_of_geo_prop_refinement(point, hjkl, point_tree2, rad_tresh, diff, Counter_check, what_shape, plotkon))
    return properties

def parallel_geo_prop_refinement(discretized_points2, hjkl, point_tree2, rad_tresh, diff, Counter_check, what_shape, plotkon, chunk_size):
    chunks = [discretized_points2[i:i + chunk_size] for i in range(0, len(discretized_points2), chunk_size)]
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, chunk, hjkl, point_tree2, rad_tresh, diff, Counter_check, what_shape, plotkon) for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
    return results


def compute_loop_of_geo(aaaac, i, point_tree, point_tree2, walls_check, walls, hjkl, rad_tresh, perc, WandL, Arounds, what_shape, plotkon, Counter_check, Counter_check_beam,  Counter_check_walls, Counter_check_column, Counter_check_add_tilted, all, proj):
    ped1xyz=aaaac[i,[3,4,5,16]]
    ped2xyz=aaaac[i,[6,7,8,17]]
    diff=ped1xyz-ped2xyz

    mids=(ped1xyz+ped2xyz)/2

    around= point_tree.data[point_tree.query_ball_point([mids[0],mids[1],mids[2]], hjkl)]
    around=np.array(around)
    Counter_check=Counter_check+1

    if walls_check:
        # Compare walls with the columns of around
        # Extract the first three columns of around
        pcd_first_three = around.copy()

        # Find the rows in around that are in walls
        maskW = np.all(np.isin(pcd_first_three, walls), axis=1)
    else:
        maskW=[]

    len_of_element=np.sqrt(diff[0]**2+diff[1]**2+diff[2]**2)
    sin_ang_of_element=np.sqrt(diff[0]**2+diff[1]**2)/len_of_element

    if sin_ang_of_element > 0.34: # 0.34 is Sin(20 degree)
        if aaaac[i, 15]==404:
            Eid=404
            Eid2=405
            Counter_check_beam += 1
        else:
            if np.any(maskW):
                Eid=300
                Eid2=Eid
                Counter_check_walls +=1
            if not np.any(maskW):
                if  sin_ang_of_element < 0.70 : # 0.71 is Sin(45 degree)
                    Eid=200
                    Eid2=Eid
                    Counter_check_add_tilted += 1
                else:
                    Counter_check_beam += 1
                if abs(diff[0]) > abs(diff[1]):
                    Eid=100
                    Eid2=Eid
                if abs(diff[0]) < abs(diff[1]):
                    Eid=10
                    Eid2=Eid

        around2 = np.hstack((around, np.ones([around.shape[0], 1]) * Eid2))
        all = np.append(around2, all, axis=0)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            around_proj_list = list(executor.map(lambda p: project_parallel(p, np.array([mids[0], mids[1], mids[2]]), np.array([diff[0], diff[1], diff[2]])), around))

        around_proj = np.array(around_proj_list)

        properties=[]
        discretized_points2 = discretize_line(ped1xyz[0:3], ped2xyz[0:3], 10)
        properties = parallel_geo_prop_refinement(discretized_points2, hjkl, point_tree2, rad_tresh, diff, Counter_check, what_shape, plotkon, chunk_size=int(np.floor(my_chunk*len(discretized_points2))))
        properties = np.array(properties)
        properties=remove_zeros_and_nan(properties)
        properties_val = find_shared_indices_above_percentile(properties, 4, 5, perc)
        AreA, Ix, Iy, Jx_y, length, width = properties_val[0]

        WandL.append([width,length,i, ped1xyz[0], ped1xyz[1], ped1xyz[2], ped2xyz[0], ped2xyz[1], ped2xyz[2], diff[0], diff[1], diff[2], Ix, Iy, Jx_y, Eid, ped1xyz[3], ped2xyz[3], AreA, len_of_element])
        
        around2 = np.hstack((around_proj, np.ones([around_proj.shape[0], 1]) * Eid2))
        proj = np.append(around2, proj, axis=0)
        Arounds.append([all, proj, i])
    elif sin_ang_of_element < 0.34: # 0.34 is Sin(20 degree)
        if aaaac[i, 15]==404:
            Eid=404
            Eid2=405
            Counter_check_column +=1
        else:
            if np.any(maskW):
                Eid=300
                Eid2=Eid
                Counter_check_walls +=1
            else:
                Eid=1
                Eid2=Eid
                Counter_check_column +=1
        
        around2 = np.hstack((around, np.ones([around.shape[0], 1]) * Eid2))
        all = np.append(around2, all, axis=0)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            around_proj_list = list(executor.map(lambda p: project_parallel(p, np.array([mids[0], mids[1], mids[2]]), np.array([diff[0], diff[1], diff[2]])), around))

        around_proj = np.array(around_proj_list)
        
        properties=[]
        discretized_points2 = discretize_line(ped1xyz[0:3], ped2xyz[0:3], 10)
        properties = parallel_geo_prop_refinement(discretized_points2, hjkl, point_tree2, rad_tresh, diff, Counter_check, what_shape, plotkon, chunk_size=int(np.floor(my_chunk*len(discretized_points2))))
        properties = np.array(properties)
        properties=remove_zeros_and_nan(properties)
        properties_val = find_shared_indices_above_percentile(properties, 4, 5, perc)
        AreA, Ix, Iy, Jx_y, length, width = properties_val[0]

        WandL.append([width,length,i, ped1xyz[0], ped1xyz[1], ped1xyz[2], ped2xyz[0], ped2xyz[1], ped2xyz[2], diff[0], diff[1], diff[2], Ix, Iy, Jx_y, Eid, ped1xyz[3], ped2xyz[3], AreA, len_of_element])
        
        around2 = np.hstack((around_proj, np.ones([around_proj.shape[0], 1]) * Eid2))
        proj = np.append(around2, proj, axis=0)
        Arounds.append([all, proj, i])
    return WandL,  proj, all, Arounds, Counter_check, Counter_check_beam,  Counter_check_walls, Counter_check_column, Counter_check_add_tilted


def geo_prop_refinement(pcd2, pcdtocross, aaaac, walls, plotkon, what_shape, perc, whatT, depth):
    if whatT==1:
        hjkl=0.5
        rad_tresh=hjkl
        if not perc:
            perc=125
    if whatT==0:
        hjkl=0.6
        rad_tresh=1 # from 1.5
        if not perc:
            perc=1
    if whatT==2:
        hjkl=1
        rad_tresh=1.5
        pcdtocross=pcdtocross[pcdtocross[:, 2]<pcdtocross[:, 2].max()-depth]
        if not perc:
            perc=10
    aaaac=np.array(aaaac)

    point_tree2 = spatial.cKDTree(pcd2.copy())
    point_tree = spatial.cKDTree(pcdtocross.copy())

    WandL=[]
    Arounds=[]
    Counter_check=0
    all = np.zeros((2, 4))
    proj = np.zeros((2, 4))
    WandL.append([0] * 20)

    Counter_check=0
    Counter_check_beam=0
    Counter_check_walls=0
    Counter_check_column=0
    Counter_check_add_tilted=0
    process=1
    process_100=len(aaaac)

    walls_check = walls.shape[0] > 0 


    for i in range(len(aaaac)):
        WandL,  proj, all, Arounds, Counter_check, Counter_check_beam,  Counter_check_walls, Counter_check_column, Counter_check_add_tilted= compute_loop_of_geo(aaaac, i, point_tree, point_tree2, walls_check, walls, hjkl, rad_tresh, perc, WandL, Arounds, what_shape, plotkon, Counter_check, Counter_check_beam,  Counter_check_walls, Counter_check_column, Counter_check_add_tilted, all, proj)
        print('{0}'.format(np.ceil(100*(process/process_100))), 'percent is done', end="\r")
        process +=1
            
    print("Correct Tilted Elements Number= ", Counter_check)
    print("Correct Tilted Beam Number= ", Counter_check_beam+Counter_check_walls)
    print("Correct Tilted Column Number= ", Counter_check_column+Counter_check_walls)
    print("Correct Additional Tilted Elements Number= ", Counter_check_add_tilted)

    return  WandL[1:], proj[2:], all[2:], Arounds


def find_nodes_under_conditions2(WandL, start, end, threshold, id, zrang):
    nodes3 = (np.array(WandL)[:, 3:6]+np.array(WandL)[:, 6:9])/2
    id3=np.array(WandL)[:, 15]
    
    x1, y1, z1 = start
    x2, y2, z2 = end
    
    x_ref, y_ref, z_ref = (x1+x2)/2, (y1+y2)/2, (z1+z2)/2

    result_nodes = []
    jacs5=0
    if (not(id==10 or id==100 or id==404)) and z_ref > zrang[0]:
        for node in nodes3:
            x3, y3, z3 = node
            if (z_ref-z3) > threshold  and np.isclose(x3, x_ref, atol=threshold) and np.isclose(y3, y_ref, atol=threshold) and (id3[jacs5]==10 or id3[jacs5]==100 or id3[jacs5]==404):
                result_nodes.append(node)
    else:
        result_nodes.append([x_ref, y_ref, z_ref])
    jacs5 +=1
    
    return np.array(result_nodes)

# Function to find midpoints that meet the conditions
def find_midpoints(lines, Ind_c, zrang, xrang, yrang, whatT, tolerance=0.01):
    midpoints = []
    arrays_to_remove = []
    new_lines = lines.copy()  # To store the updated lines
    # New column with default values (e.g., zeros)
    new_column = np.zeros((new_lines.shape[0], 1))

    # Append the new column to the lines array
    new_lines = np.hstack((new_lines, new_column))
    for i in Ind_c:
        line = lines[i, :]
        start = line[3:6]
        end = line[6:9]
        midpoint = (start + end) / 2
        z_mid1 = midpoint[2]
        for j in Ind_c:
            if i != j:
                other_line = lines[j, :]
                other_start = other_line[3:6]
                other_end = other_line[6:9]
                other_midpoint = (other_start + other_end) / 2
                z_mid2 = other_midpoint[2]
                if whatT==2:
                    condition=(z_mid2 < z_mid1 and np.allclose(midpoint[:2], other_midpoint[:2], atol=tolerance) and z_mid2>=zrang[-2] and other_midpoint[0]>xrang[0] and other_midpoint[0]<xrang[-1] and other_midpoint[1]>yrang[0] and other_midpoint[1]<yrang[-1])
                else:
                    condition=(z_mid2 < z_mid1 and np.allclose(midpoint[:2], other_midpoint[:2], atol=tolerance) and z_mid2>=zrang[-2] and other_midpoint[1]>yrang[0] and other_midpoint[1]<yrang[-1])                    
                if condition:
                    new_start = start
                    new_end = end
                    new_diff= new_start - new_end
                    width=(line[0]+other_line[0])/2
                    length=(line[1]+other_line[1])/2
                    iiii=line[2]
                    ped1xyz0=new_start[0]
                    ped1xyz1=new_start[1]
                    ped1xyz2=new_start[2]
                    ped2xyz0=new_end[0]
                    ped2xyz1=new_end[1]
                    ped2xyz2=new_end[2]
                    diff0=new_diff[0]
                    diff1=new_diff[1]
                    diff2=new_diff[2]
                    Ix=(width*(length**3))/12.0
                    Iy=((width**3)*length)/12.0
                    Jx_y=Ix + Iy
                    Eid=line[15]
                    Eid2=405
                    ped1xyz3=line[16]
                    ped2xyz3=line[17]
                    area=width*length
                    len_of_element=np.linalg.norm(new_diff)
                    new_line_add=np.array([width, length, iiii, ped1xyz0, ped1xyz1, ped1xyz2, ped2xyz0, ped2xyz1, ped2xyz2, diff0, diff1, diff2, Ix, Iy, Jx_y, Eid, ped1xyz3, ped2xyz3, area, len_of_element, Eid2])
            
                    # Add the new merged line
                    new_lines = np.vstack([new_lines, new_line_add])
                    midpoints.append(midpoint.tolist()+other_midpoint.tolist()+[i, j])
                    # print(new_lines.shape, new_line_add.shape)
                    arrays_to_remove.append(i)
                    arrays_to_remove.append(j)
                    break
        # print(arrays_to_remove)
    new_lines = np.delete(new_lines, arrays_to_remove, axis=0)
    array_mid=np.array(midpoints)
    return array_mid, new_lines

def refine_deep_elements(WandL, whatT, vis_it, zrang, xrang, yrang):
    if whatT==2 or whatT==0:
        list_checker=[]
        for i_check in range(len(WandL)):
            acheck=find_nodes_under_conditions2(WandL, WandL[i_check][3:6], WandL[i_check][6:9], 0.05, WandL[i_check][15], zrang)
            if len(acheck):
                # print(i_check)
                list_checker.append([i_check, i_check, (WandL[i_check][0]), (WandL[i_check][1])])
        np_ind_s_e=np.array(list_checker, dtype=int)
        choosen_ind=np_ind_s_e[:, 0].tolist()
        midpoints, new_array_wand= find_midpoints(np.array(WandL), choosen_ind, zrang, xrang, yrang, whatT)
        if vis_it:
            # Visualize the edges
            fig = plt.figure(figsize=(10, 10))
            axg = fig.add_subplot(111, projection='3d')

            avvr=np.array(WandL)

            for edge in avvr:
                if not (edge[0]==0 and edge[1]==0):
                    x = [edge[3], edge[6]]
                    y = [edge[4], edge[7]]
                    z = [edge[5], edge[8]]
                    axg.plot(x, y, z, 'b-',linewidth=2)

            x = [edge[3], edge[6]]
            y = [edge[4], edge[7]]
            z = [edge[5], edge[8]]
            axg.plot(x, y, z, 'b-', label="Refined Element",linewidth=2)

            for edge in avvr:
                if not (edge[0]==0 and edge[1]==0):
                    x = [edge[3], edge[6]]
                    y = [edge[4], edge[7]]
                    z = [edge[5], edge[8]]
                    axg.scatter(x, y, z, 'b-', c="red", s=50)

            x = [edge[3], edge[6]]
            y = [edge[4], edge[7]]
            z = [edge[5], edge[8]]
            axg.scatter(x, y, z, 'b-', c="red", label="Refined Node", s=50)

            avvr=new_array_wand

            itext=0
            for edge in avvr:
                if not (edge[0] == 0 and edge[1] == 0):
                    x = [edge[3], edge[6]]
                    y = [edge[4], edge[7]]
                    z = [edge[5], edge[8]]
                    axg.plot(x, y, z, 'r-', linewidth=5)
                    mid_x = (x[0] + x[1]) / 2
                    mid_y = (y[0] + y[1]) / 2
                    mid_z = (z[0] + z[1]) / 2
                    itext +=1

            x = [edge[3], edge[6]]
            y = [edge[4], edge[7]]
            z = [edge[5], edge[8]]
            axg.plot(x, y, z, 'r-', label="Remained Element",linewidth=5)

            for edge in avvr:
                if not (edge[0]==0 and edge[1]==0):
                    x = [edge[3], edge[6]]
                    y = [edge[4], edge[7]]
                    z = [edge[5], edge[8]]
                    axg.scatter(x, y, z, 'b-', c="black", s=100)

            x = [edge[3], edge[6]]
            y = [edge[4], edge[7]]
            z = [edge[5], edge[8]]
            axg.scatter(x, y, z, 'b-', c="black", label="Node", s=100)

            # Set labels
            axg.set_xlabel('X (m)')
            axg.set_ylabel('Y (m)')
            axg.set_zlabel('Z (m)')

            axg.set_box_aspect(asr)
            axg.legend(bbox_to_anchor=(1.2, 1.0), loc='upper left')

            # Set title
            axg.set_title("Refined Nodes and Elements")

        return new_array_wand.tolist()
    else:
        # Add the new column to each line in the list
        lines_with_new_column = [line + [0] for line in WandL]
        return lines_with_new_column

def Plot_after_cross_section_extraction(aaaac, aaaac2, WandL, proj, all, asr, aind=15):
    # Plot after cross-section extraction
    proj=proj[abs(proj[:,0])+abs(proj[:,1])+abs(proj[:,2])>0]
    all=all[abs(all[:,0]+all[:,1]+all[:,2])>0]
    pcdtocrossC=proj.copy()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for edge in aaaac:
        x = [edge[0, 0], edge[1, 0]]
        y = [edge[0, 1], edge[1, 1]]
        z = [edge[0, 2], edge[1, 2]]
        ax.plot(x, y, z, 'b-',linewidth=2)

    x = [edge[0, 0], edge[1, 0]]
    y = [edge[0, 1], edge[1, 1]]
    z = [edge[0, 2], edge[1, 2]]
    ax.plot(x, y, z, 'b-', label="Corrected Element",linewidth=3)

    XX=pcdtocrossC[:,0]
    YY=pcdtocrossC[:,1]
    ZZ=pcdtocrossC[:,2]
    mask = pcdtocrossC[:, 3] == 1
    mask2 = (pcdtocrossC[:, 3] == 100) | (pcdtocrossC[:, 3] == 10)
    mask3 = pcdtocrossC[:, 3] == 300
    mask4 = pcdtocrossC[:, 3] == 200


    ax.scatter(XX[mask2], YY[mask2], ZZ[mask2], c='olive', alpha=0.05, s=50)
    ax.scatter(XX[mask2][0], YY[mask2][0], ZZ[mask2][0], c='olive', label="Beam", alpha=1)
    ax.scatter(XX[mask], YY[mask], ZZ[mask], c='green', alpha=0.05, s=50)
    ax.scatter(XX[mask][0], YY[mask][0], ZZ[mask][0], c='green', label="Column", alpha=1)

    if any(mask3):
        ax.scatter(XX[mask3], YY[mask3], ZZ[mask3], c='purple', alpha=0.05, s=50)
        ax.scatter(XX[mask3][0], YY[mask3][0], ZZ[mask3][0], c='purple', label="Wall", alpha=1)

    if any(mask4):
        ax.scatter(XX[mask4], YY[mask4], ZZ[mask4], c='black', alpha=0.05, s=50)
        ax.scatter(XX[mask3][0], YY[mask3][0], ZZ[mask3][0], c='black', label="Additional Tilted Element", alpha=1)
    
    mask5 = pcdtocrossC[:, 3] == 405
    if any(mask5):
        ax.scatter(XX[mask5], YY[mask5], ZZ[mask5], c='m', alpha=0.05, s=50)
        ax.scatter(XX[mask5][0], YY[mask5][0], ZZ[mask5][0], c='m', label="Deep Beam", alpha=1)

    # Set labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_box_aspect(asr)

    ax.legend(bbox_to_anchor=(1.2, 1.0), loc='upper left')

    # Set title
    ax.set_title("Cross Sections - Corrected Elements")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    
    for edge in aaaac2:
        x = [edge[0, 0], edge[1, 0]]
        y = [edge[0, 1], edge[1, 1]]
        z = [edge[0, 2], edge[1, 2]]
        ax.plot(x, y, z, 'b-',linewidth=0.5)

    x = [edge[0, 0], edge[1, 0]]
    y = [edge[0, 1], edge[1, 1]]
    z = [edge[0, 2], edge[1, 2]]
    ax.plot(x, y, z, 'b-', label="Additional Tilted Element",linewidth=0.5)

    # Set labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')


    ax.legend(bbox_to_anchor=(1.2, 1.0), loc='upper left')

    # Set title
    ax.set_title("Cross Sections - Additional Tilted Elements")
    ax.set_box_aspect(asr)
    plt.show()


    avvr=np.array(WandL)
    print(len(avvr))
    pcdtocrossC=all.copy()
    XX=pcdtocrossC[:,0]
    YY=pcdtocrossC[:,1]
    ZZ=pcdtocrossC[:,2]
    mask = pcdtocrossC[:, 3] == 1
    mask2 = (pcdtocrossC[:, 3] == 100) | (pcdtocrossC[:, 3] == 10)
    mask3 = pcdtocrossC[:, 3] == 300
    mask4 = pcdtocrossC[:, 3] == 200

    # Visualize the edges
    fig = plt.figure(figsize=(10, 10))
    axg = fig.add_subplot(111, projection='3d')

    avvr=np.array(WandL)
    wall_elements=[]
    wall_elements_i=[]
    i=0
    for edge in avvr:
        if not (edge[0]==0 and edge[1]==0):
            if edge[15]==300:
                wall_elements.append(avvr[i,:])
                wall_elements_i.append(i)
        i +=1
    if len(wall_elements_i):
        atestt=np.array(wall_elements)
        arr=atestt[:, [3,4,6,7]].copy()

        # Find unique rows and their indices
        _, indices, counts = np.unique(arr, axis=0, return_index=True, return_counts=True)

        # Identify duplicate rows
        duplicate_indices = np.where(counts > 1)[0]

        # Get the indices of all occurrences of duplicate rows
        all_duplicate_indices = [np.where((arr == arr[idx]).all(axis=1))[0] for idx in indices[duplicate_indices]]

        delthem=[]
        for indices in all_duplicate_indices:
            startL=[]
            endL=[]
            for i in indices:
                startL.append(atestt[i, 3:6])
                endL.append(atestt[i, 6:9])
                delthem.append(i)
            mins=np.min(startL, axis=0)
            maxs=np.max(endL, axis=0)
            ind_plane=np.where(maxs-mins<0.2)[0]
            if len(ind_plane)!=1:
                means=np.mean(atestt[indices, :], axis=0)
                means[3:6]=mins
                means[6:9]=maxs
                means[15]=300
                means[20]=0
                atestt=np.append(atestt, means.reshape(1, atestt.shape[1]), axis=0)
            elif ind_plane==0:
                means=np.mean(atestt[indices, :], axis=0)
                means[3]=mins[0]
                means[4]=mins[1]
                means[5]=maxs[2]

                means[6]=mins[0]
                means[7]=maxs[1]
                means[8]=maxs[2]

                means[15]=300
                means[20]=0
                atestt=np.append(atestt, means.reshape(1, atestt.shape[1]), axis=0)

                means=np.mean(atestt[indices, :], axis=0)
                means[3]=mins[0]
                means[4]=mins[1]
                means[5]=mins[2]

                means[6]=mins[0]
                means[7]=maxs[1]
                means[8]=mins[2]

                means[15]=300
                means[20]=0
                atestt=np.append(atestt, means.reshape(1, atestt.shape[1]), axis=0)      
        

        atestt = np.delete(atestt, delthem, axis=0)
        avvr34=avvr.copy()
        avvr34 = np.delete(avvr34, wall_elements_i, axis=0)
        avvr34 = np.append(avvr34, atestt, axis=0)
    else:
        avvr34=np.array(WandL)
    
    avvr=avvr34
    for edge in avvr:
        if not (edge[0]==0 and edge[1]==0):
            x = [edge[3], edge[6]]
            y = [edge[4], edge[7]]
            z = [edge[5], edge[8]]
            if edge[aind]==405:
                axg.plot(x, y, z, c='m',linewidth=2)
            elif edge[15]==300:
                axg.plot(x, y, z, c='c',linewidth=2)
            else:
                axg.plot(x, y, z, c='b',linewidth=2)

    x = [edge[3], edge[6]]
    y = [edge[4], edge[7]]
    z = [edge[5], edge[8]]
    if edge[aind]==405:
        axg.plot(x, y, z, c='m', label="Remained Deep Beam",linewidth=2)
    elif edge[15]==300:
        axg.plot(x, y, z, c='c', label="Wall",linewidth=2)
    else:
        axg.plot(x, y, z, c='b', label="Remained Element",linewidth=2)

    for edge in avvr:
        if not (edge[0]==0 and edge[1]==0):
            x = [edge[3], edge[6]]
            y = [edge[4], edge[7]]
            z = [edge[5], edge[8]]
            if edge[aind]==405:
                axg.scatter(x, y, z, 'm-', c="m", s=1)
            else:
                axg.scatter(x, y, z, 'b-', c="red", s=50)

    x = [edge[3], edge[6]]
    y = [edge[4], edge[7]]
    z = [edge[5], edge[8]]
    if edge[aind]==405:
        axg.scatter(x, y, z, 'm-', c="m", label="Deep Beam Node", s=1)
    else:
        axg.scatter(x, y, z, 'b-', c="red", label="Node", s=50)
    # An optional plot of the PCD
    axg.scatter(XX[mask2], YY[mask2], ZZ[mask2], c='olive', alpha=0.05, s=50)
    axg.scatter(XX[mask2][0], YY[mask2][0], ZZ[mask2][0], c='olive', label="Beam", alpha=1)
    axg.scatter(XX[mask], YY[mask], ZZ[mask], c='green', alpha=0.05, s=50)
    axg.scatter(XX[mask][0], YY[mask][0], ZZ[mask][0], c='green', label="Column", alpha=1)

    if any(mask3):
        axg.scatter(XX[mask3], YY[mask3], ZZ[mask3], c='purple', alpha=0.05, s=50)
        axg.scatter(XX[mask3][0], YY[mask3][0], ZZ[mask3][0], c='purple', label="Wall", alpha=1)

    if any(mask4):
        axg.scatter(XX[mask4], YY[mask4], ZZ[mask4], c='black', alpha=0.05, s=50)
        axg.scatter(XX[mask3][0], YY[mask3][0], ZZ[mask3][0], c='black', label="Additional Tilted Element", alpha=1)
    
    mask5 = pcdtocrossC[:, 3] == 405
    if any(mask5):
        axg.scatter(XX[mask5], YY[mask5], ZZ[mask5], c='m', alpha=0.05, s=50)
        axg.scatter(XX[mask5][0], YY[mask5][0], ZZ[mask5][0], c='m', label="Deep Beam", alpha=1)


    # Set labels
    axg.set_xlabel('X (m)')
    axg.set_ylabel('Y (m)')
    axg.set_zlabel('Z (m)')

    axg.set_box_aspect(asr)
    axg.legend(bbox_to_anchor=(1.2, 1.0), loc='upper left')

    # Set title
    axg.set_title("Refined Nodes and Elements")

    plt.show()


    # Visualize the edges
    fig = plt.figure(figsize=(10, 10))
    axg = fig.add_subplot(111, projection='3d')

    avvr=avvr34
    for edge in avvr:
        if not (edge[0]==0 and edge[1]==0):
            x = [edge[3], edge[6]]
            y = [edge[4], edge[7]]
            z = [edge[5], edge[8]]
            if edge[aind]==405:
                axg.plot(x, y, z, c='m',linewidth=2)
            elif edge[15]==300:
                axg.plot(x, y, z, c='c',linewidth=2)
            else:
                axg.plot(x, y, z, c='b',linewidth=2)

    x = [edge[3], edge[6]]
    y = [edge[4], edge[7]]
    z = [edge[5], edge[8]]
    if edge[aind]==405:
        axg.plot(x, y, z, c='m', label="Remained Deep Beam",linewidth=2)
    elif edge[15]==300:
        axg.plot(x, y, z, c='c', label="Wall",linewidth=2)
    else:
        axg.plot(x, y, z, c='b', label="Remained Element",linewidth=2)

    for edge in avvr:
        if not (edge[0]==0 and edge[1]==0):
            x = [edge[3], edge[6]]
            y = [edge[4], edge[7]]
            z = [edge[5], edge[8]]
            if edge[aind]==405:
                axg.scatter(x, y, z, 'm-', c="m", s=1)
            else:
                axg.scatter(x, y, z, 'b-', c="red", s=50)

    x = [edge[3], edge[6]]
    y = [edge[4], edge[7]]
    z = [edge[5], edge[8]]
    if edge[aind]==405:
        axg.scatter(x, y, z, 'm-', c="m", label="Deep Beam Node", s=1)
    else:
        axg.scatter(x, y, z, 'b-', c="red", label="Node", s=50)

    # Set labels
    axg.set_xlabel('X (m)')
    axg.set_ylabel('Y (m)')
    axg.set_zlabel('Z (m)')

    axg.set_box_aspect(asr)
    axg.legend(bbox_to_anchor=(1.2, 1.0), loc='upper left')

    # Set title
    axg.set_title("Refined Nodes and Elements")

    plt.show()
    return avvr34.tolist()

def plot_MINI_BIM(walls, Ceiling2, WandL, all, asr, MODE):
    all=all[all[:,0]+all[:,1]+all[:,2]>0]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    if walls.shape[0]:
        # Visualize Wall
        XXsw=walls[:,0]
        YYsw=walls[:,1]
        ZZsw=walls[:,2]

        ax.scatter(XXsw, YYsw, ZZsw, c='purple', alpha=0.1, s=1)
        ax.scatter(XXsw[-1], YYsw[-1], ZZsw[-1], c='purple', label="Wall", alpha=1)

    # Visualize Ceiling

    XXs=Ceiling2[:,0]
    YYs=Ceiling2[:,1]
    ZZs=Ceiling2[:,2]

    ax.scatter(XXs, YYs, ZZs, c='blue', alpha=0.01, s=1)
    ax.scatter(XXs[-1], YYs[-1], ZZs[-1], c='blue', label="Ceiling", alpha=1)

    avvr=np.array(WandL)
    pcdtocrossC=all.copy()

    if walls.shape[0]:
        # Compare walls with the first three columns of pcdtocrossC
        # Extract the first three columns of pcdtocrossC
        pcd_first_three = pcdtocrossC[:, :3]

        # Find the rows in pcdtocrossC that are not in walls
        mask = np.all(np.isin(pcd_first_three, walls), axis=1)
        pcdtocrossC = pcdtocrossC[~mask]

    XX=pcdtocrossC[:,0]
    YY=pcdtocrossC[:,1]
    ZZ=pcdtocrossC[:,2]
    mask = pcdtocrossC[:, 3] == 1

    ax.scatter(XX[~mask], YY[~mask], ZZ[~mask], c='olive', alpha=0.05, s=5)
    ax.scatter(XX[0], YY[0], ZZ[0], c='olive', label="Beam", alpha=1)
    Beam=np.vstack((XX[mask],YY[mask],ZZ[mask]))

    ax.scatter(XX[mask], YY[mask], ZZ[mask], c='green', alpha=0.05, s=5)
    ax.scatter(XX[0], YY[0], ZZ[0], c='green', label="Column", alpha=1)
    Column=np.vstack((XX[~mask], YY[~mask], ZZ[~mask]))

    if MODE:
        avvr=np.array(WandL)

        for edge in avvr:
            if not (edge[0]==0 and edge[1]==0):
                x = [edge[3], edge[6]]
                y = [edge[4], edge[7]]
                z = [edge[5], edge[8]]
                ax.plot(x, y, z, c="yellow", linewidth=3)

        x = [edge[3], edge[6]]
        y = [edge[4], edge[7]]
        z = [edge[5], edge[8]]
        ax.plot(x, y, z, c="yellow", label="Remained Element",linewidth=3)

        for edge in avvr:
            if not (edge[0]==0 and edge[1]==0):
                x = [edge[3], edge[6]]
                y = [edge[4], edge[7]]
                z = [edge[5], edge[8]]
                ax.scatter(x, y, z, c="red", s=50)

        x = [edge[3], edge[6]]
        y = [edge[4], edge[7]]
        z = [edge[5], edge[8]]
        ax.scatter(x, y, z, c="red", label="Node", s=50)

    # Set labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_box_aspect(asr)
    ax.legend(bbox_to_anchor=(1.2, 1.0), loc='upper left')

    # Set title
    ax.set_title("Semantic Segmentation")


    plt.show()

def remove_zeros_and_nan(array):
    # Check for finite numbers greater than 0
    valid_rows = np.all(np.isfinite(array) & (array > 0), axis=1)

    # Replace invalid rows with the mean of valid rows
    arrayout=array[valid_rows]

    return arrayout

def find_natches(in_data, zrange, thre, w, l, gt_mode, mod=1):
    ''' this function computes average geometrical peroperties'''
    datta_store=[]
    for ceiling_i in zrange:
        if mod==1:
            mask=np.where(np.abs(in_data[:, 5]-ceiling_i)<0.5)[0]
        else:
            mask=range(len(in_data))

        indicator=0
        xdata=in_data[mask, indicator]
        g=remove_outliers(xdata, thre, 1)
        if gt_mode and w!=0:
            mean_w=w
        else:
            mean_w=np.mean(g)
        in_data[mask, indicator]=mean_w
        
        indicator=1
        xdata=in_data[mask, indicator]
        g=remove_outliers(xdata, thre, 1)
        if gt_mode and l!=0:
            mean_l=l
        else:
            mean_l=np.mean(g)
        in_data[mask, indicator]=mean_l

        indicator=12
        mean_Ix=(mean_w*mean_l**3)/12
        in_data[mask, indicator]=mean_Ix

        indicator=13
        mean_Iy=(mean_l*mean_w**3)/12
        in_data[mask, indicator]=mean_Iy

        indicator=14
        mean_J=mean_Iy+mean_Ix
        in_data[mask, indicator]=mean_J

        indicator=18
        mean_area=mean_w*mean_l
        in_data[mask, indicator]=mean_area
    return in_data

def clean_properties(avvr2, zrange, thre, gt_mode):
    avvr3=avvr2.copy()

    avvr3X = avvr3[(avvr3[:, 15] == 100) & (avvr3[:, 20] == 0)]
    avvr3Y = avvr3[(avvr3[:, 15] == 10) & (avvr3[:, 20] == 0)]
    avvr3Z = avvr3[(avvr3[:, 15] == 1) & (avvr3[:, 20] == 0)]
    avvr3T = avvr3[(avvr3[:, 15] == 200) & (avvr3[:, 20] == 0)]
    avvr3W = avvr3[(avvr3[:, 15] == 300) & (avvr3[:, 20] == 0)]
    avvr3C = avvr3[avvr3[:, 20] == 405]


    avvr3X=find_natches(avvr3X, zrange, thre, 0.27305, 0.8763, gt_mode)
    avvr3Y=find_natches(avvr3Y, zrange, thre, 0.27305, 0.8763, gt_mode)
    avvr3Z=find_natches(avvr3Z, zrange, thre, 0.27305, 0.27305, gt_mode, 0)

    if  len(avvr3T):
        avvr3T=find_natches(avvr3T, zrange, thre, 0.27305, 0.8763, gt_mode)
    if  len(avvr3W):
        avvr3W=find_natches(avvr3W, zrange, thre, 0, 0, gt_mode, 0)
    if  len(avvr3C):
        avvr3C=find_natches(avvr3C, zrange, thre, 0.27305, 0.8763, gt_mode)

    avvr4=np.vstack((avvr3X, avvr3Y, avvr3Z))

    if  len(avvr3T): avvr4=np.vstack((avvr4, avvr3T))
    if  len(avvr3W): avvr4=np.vstack((avvr4, avvr3W))
    if  len(avvr3C): avvr4=np.vstack((avvr4, avvr3C))

    return avvr4

def opensees_preparation(nodes2, WandL, xrang, yrang, zrang, thre, gt_mode=0):
    # Preparing data for the OpenSeesPy
    
    avvr = np.array(WandL)
    avvr2 = avvr[avvr[:, 15] != 0]
    nnn = np.array(nodes2.copy())
    output_to_opensees = clean_properties(avvr2, zrang, thre, gt_mode)
    output_to_opensees_copy = output_to_opensees.copy()
    threshold = 0.36
    A2_slice = nnn[:, 0:3]
    my_elm = []

    for k in range(len(output_to_opensees)):
        A1_slice1 = output_to_opensees_copy[k, 3:6]
        A1_slice2 = output_to_opensees_copy[k, 6:9]
        
        close_ind_1 = np.where(np.all(np.abs(A2_slice - A1_slice1) <= threshold, axis=1))[0]
        close_ind_2 = np.where(np.all(np.abs(A2_slice - A1_slice2) <= threshold, axis=1))[0]
        
        dumpy = np.hstack((output_to_opensees[k, :], close_ind_1, close_ind_2, k))
        my_elm.append(dumpy)
    
    my_elm = np.array(my_elm)

    return nnn, avvr2, [xrang, yrang, zrang], output_to_opensees, my_elm



import scipy.spatial as spatial
import matplotlib.pyplot as plt

def find_points_around_line(WandLar, targetpcd, Ceiling2, walls, Siz=(10, 10)):
    print('---------- WARNING ----------')
    print(''' Dense Semantic Segmentation
-------- Might Take --------
------- (A LONG TIME) -------
-------- To Be Done --------''')
    print('---------- WARNING ----------')

    point_tree = spatial.cKDTree(targetpcd.copy())
    num_points = 100
    Dense_mini_bim = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
    process_100=WandLar.shape[0]
    process=1
    for i in range(WandLar.shape[0]):
        diameter = 0.5 * np.sqrt(WandLar[i, 0]**2 + WandLar[i, 1]**2)
        start = np.array([WandLar[i, 3], WandLar[i, 4], WandLar[i, 5]])
        end = np.array([WandLar[i, 6], WandLar[i, 7], WandLar[i, 8]])
        discretized_points = discretize_line(start, end, num_points)
        for point in discretized_points:
            tree_ind = point_tree.query_ball_point(point, diameter)
            while not tree_ind:
                diameter *= 1.5  # Increase the diameter
                tree_ind = point_tree.query_ball_point(point, diameter)
            tree_ind = point_tree.query_ball_point(point, (diameter*1.2))
            around_points = point_tree.data[tree_ind]
            around_points = np.asarray(around_points)
            if tree_ind:
                if WandLar[i, 15] == 10 or WandLar[i, 15] == 100:
                    color_code = 0
                if WandLar[i, 15] == 200:
                    color_code = 0.125
                if WandLar[i, 15] == 1:
                    color_code = 0.25
                if WandLar[i, 15] == 300:
                    color_code = 0.5
                oneone = np.ones([around_points.shape[0], 1]) * color_code
                around_points_colored = np.hstack((around_points, oneone))
                Dense_mini_bim = np.append(around_points_colored, Dense_mini_bim, axis=0)
        print('{0}'.format(np.ceil(100*(process/process_100))), 'percent is done', end="\r")
        process +=1

    color_code=1
    oneone=np.ones([Ceiling2.shape[0],1])*color_code
    around_points_colored=np.hstack((Ceiling2,oneone))
    Dense_mini_bim=np.append(around_points_colored, Dense_mini_bim, axis=0)

    color_code=0.5
    oneone=np.ones([walls.shape[0],1])*color_code
    around_points_colored=np.hstack((walls,oneone))
    Dense_mini_bim=np.append(around_points_colored, Dense_mini_bim, axis=0)

    Dense_mini_bim=Dense_mini_bim[2:, :]

    Dense_mini_bim=Dense_mini_bim[abs(Dense_mini_bim[:,0])+abs(Dense_mini_bim[:,1])+abs(Dense_mini_bim[:,2])>0]

    print('---------- Done ----------')
    return Dense_mini_bim


import open3d as o3d
import numpy as np

def save_and_downsample_point_cloud(np_array, filename, voxel_size=0.05):
    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()

    # Assign points and colors from the NumPy array
    pcd.points = o3d.utility.Vector3dVector(np_array[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(np.repeat(np_array[:, 3:4], 3, axis=1))  # Repeat the color value for R, G, B

    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Save the point cloud to a PCD file
    o3d.io.write_point_cloud(filename, pcd)

    # Downsample the point cloud
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Convert the downsampled point cloud back to a NumPy array
    downsampled_array = np.hstack((np.asarray(downsampled_pcd.points), np.asarray(downsampled_pcd.colors)[:, 0:1]))

    return downsampled_array


# Create a 3D scatter plot using Plotly
import plotly.graph_objects as go

def plot3D(toplot, cc3=[]):
    # Set colors if there is no given color
    if len(cc3)==0: cc3=toplot[:,0]+toplot[:,1]+toplot[:,2]

    fig = go.Figure(data=[go.Scatter3d(
        x=toplot[:,0], y=toplot[:,1], z=toplot[:,2],
        mode='markers',
        marker=dict(
            size=2,
            color=cc3,  # Set color to the z values
            colorscale='Viridis',
            opacity=0.8
        )
    )])

    # Set plot labels
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title='Downsampled Point Cloud'
    )

    # Show the plot
    fig.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:  # If the vectors are parallel
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def is_point_in_box(point, box_min, box_max):
    return np.all(point >= box_min) and np.all(point <= box_max)

def filter_points_in_boxes(points, boxes):
    inside_points = []
    for point in points:
        for box in boxes:
            if is_point_in_box(point, box[0], box[1]):
                inside_points.append(point)
                break
    return np.array(inside_points)

def plot_3d_bounding_box(ax, start_point, end_point, width, length, depth, color, points, plot_pcd, num):
    # Calculate the center of the bounding box
    Cpoint=[]
    center = (np.array(start_point) + np.array(end_point)) / 2
    
    # Calculate the direction vector
    direction = np.array(end_point) - np.array(start_point)
    direction = direction / np.linalg.norm(direction)  # Normalize the direction vector
    
    # Calculate the corner points of the bounding box
    corners = np.array([
        [-width/2, -length/2, -depth/2],
        [width/2, -length/2, -depth/2],
        [width/2, length/2, -depth/2],
        [-width/2, length/2, -depth/2],
        [-width/2, -length/2, depth/2],
        [width/2, -length/2, depth/2],
        [width/2, length/2, depth/2],
        [-width/2, length/2, depth/2]
    ])
    
    # Apply rotation to corners
    default_direction = np.array([0, 0, 1])  # Default direction along Z-axis
    R = rotation_matrix_from_vectors(default_direction, direction)
    corners = np.dot(corners, R.T)
    
    # Translate corners to the center
    corners += center
    
    # Define the 12 edges of the bounding box
    edges = [
        [corners[0], corners[1]], [corners[1], corners[2]], [corners[2], corners[3]], [corners[3], corners[0]],
        [corners[4], corners[5]], [corners[5], corners[6]], [corners[6], corners[7]], [corners[7], corners[4]],
        [corners[0], corners[4]], [corners[1], corners[5]], [corners[2], corners[6]], [corners[3], corners[7]]
    ]

    # Define bounding boxes using min and max corners
    boxes = [(np.min(corners, axis=0), np.max(corners, axis=0))]
    
    if plot_pcd:
        # Filter points inside the bounding boxes
        inside_points = filter_points_in_boxes(points, boxes)
        if len(inside_points):
            ax.scatter(inside_points[:,0], inside_points[:,1], inside_points[:,2], color=color, alpha=1)
        if len(inside_points)<num:
            add_on=0.05
            while len(inside_points)<num:
                add_on +=0.05
                inside_points = filter_points_in_boxes(points, [(np.min(corners, axis=0)-add_on, np.max(corners, axis=0)+add_on)])
            ax.scatter(inside_points[:,0], inside_points[:,1], inside_points[:,2], color=color, alpha=1)
        Cpoint.append([[inside_points], [color]])
            
    
    else:
        # Plot the edges
        for edge in edges:
            ax.plot3D(*zip(*edge), color=color)
    
    return corners, Cpoint

def data_loader_v1(path, ):
    if path=='demo_data3.pcd':
        whatT=0
        voxel_size=0.04
    if path=='demo_data1.pcd':
        whatT=1
        voxel_size=0.05
    if path=='demo_data2.pcd' :
        whatT=2
        voxel_size=0.07

    pcd_o3d = o3d.io.read_point_cloud(path)
    pcd=np.asarray(pcd_o3d.points)
    pcd2=pcd.copy()
    # Set the aspect ratio to 'auto'
    asr=[max(pcd2[:,0])-min(pcd2[:,0]), max(pcd2[:,1])-min(pcd2[:,1]), max(pcd2[:,2])-min(pcd2[:,2])]
    pcd_o3d_down=pcd_o3d.voxel_down_sample(voxel_size=voxel_size)
    pcd3=np.asarray(pcd_o3d_down.points)
    #  -----------------------------------
    V_PCD=((pcd[:,0].max()-pcd[:,0].min())*(pcd[:,1].max()-pcd[:,1].min())*(pcd[:,2].max()-pcd[:,2].min()))
    PCD_Density=(pcd.shape[0])/V_PCD
    print(f'''Density of the PCD is {np.round(PCD_Density, 2)} (Points/m3)
        Volume of the PCD is {np.round(V_PCD, 2)} (m3)
        There are {pcd.shape[0]} points in the PCD''')
    
    return pcd_o3d, pcd, pcd2, asr, pcd_o3d_down, pcd3, V_PCD, PCD_Density, whatT


def skeleton_points_plt(colored_points, asr):
    # Get the coordinates of the skeleton points
    colored_points2=colored_points.copy()
    colored_points2=colored_points2[colored_points2[:,0]+colored_points2[:,1]+colored_points2[:,2]>0]

    x = colored_points2[:, 0]
    y = colored_points2[:, 1]
    z = colored_points2[:, 2]
    c = colored_points2[:, 3]

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    text='Structure Slices'
    # text='Z Slices'
    ax.scatter(x[c==0], y[c==0], z[c==0], zdir='z', c='red', alpha=0.1, marker='x')
    ax.scatter(x[0], y[0], z[0], zdir='z', c='red', label='X Slices', alpha=1, marker='x')
    ax.scatter(x[c==1], y[c==1], z[c==1], zdir='z', c='green', alpha=0.1, marker='o')
    ax.scatter(x[0], y[0], z[0], zdir='z', c='green', label='Y Slices', alpha=1, marker='o')
    ax.scatter(x[c==2], y[c==2], z[c==2], zdir='z', c='blue', alpha=0.01, marker='^')
    ax.scatter(x[0], y[0], z[0], zdir='z', c='blue', label='Z Slices', alpha=1, marker='^')


    ax.legend(bbox_to_anchor=(1.2, 1.0), loc='upper left')

    # Set labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_box_aspect(asr)
    # Set title
    ax.set_title(text)

    # Show the plot
    plt.show()
import numpy as np
import cv2
import math
import sys


############################################################
#
#                       KMEANS
#
############################################################

# implement distance metric - e.g. squared distances between pixels
def distance(cluster_centroid, point):
    #euclidean
    return np.linalg.norm(point-cluster_centroid)

    # YOUR CODE HERE

# k-means works in 3 steps
# 1. initialize
# 2. assign each data element to current mean (cluster center)
# 3. update mean
# then iterate between 2 and 3 until convergence, i.e. until ~smaller than 5% change rate in the error

def update_mean(img, clustermask, current_cluster_centers, cluster_dict):
    """This function should compute the new cluster center, i.e. numcluster mean colors"""
    unique, counts = np.unique(clustermask, return_counts=True)
    # stores the number of corresponding points belonging to each of the k clusters
    cluster_count = dict(zip(unique, counts))
    print("Cluster distribution before updating mean: {}\n\n".format(cluster_count))
    for x in range(clustermask.shape[0]):
        for y in range(clustermask.shape[1]):
            cvalue = clustermask[x][y][0]
            if cvalue not in cluster_dict:
                cluster_dict[cvalue] = np.zeros(3)
            else:
                cluster_dict[cvalue] = np.add(cluster_dict[cvalue], img[x][y])
    for i,c in enumerate(current_cluster_centers):
        # divide sum of all points belonging to this cluster by their count ( mean )
        current_cluster_centers[i] = cluster_dict[i] / cluster_count[i]
    # YOUR CODE HERE


def assign_to_current_mean(img, result, clustermask, current_cluster_centers):
    """The function expects the img, the resulting image and a clustermask.
    After each call the pixels in result should contain a cluster_color corresponding to the cluster
    it is assigned to. clustermask contains the cluster id (int [0...num_clusters]
    Return: the overall error (distance) for all pixels to there closest cluster center (mindistance px - cluster center).
    """
    overall_dist = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            distance_to_cluster =  sys.float_info.max
            for index, cc in enumerate(current_cluster_centers):
                d = distance(cc, img[i][j])
                if d < distance_to_cluster:
                    distance_to_cluster = d
                    result[i][j] = cc
                    clustermask[i][j] = index
            overall_dist += distance_to_cluster
    return overall_dist


def initialize(img, k, current_cluster_centers):
    """inittialize the current_cluster_centers array for each cluster with a random pixel position"""
    # YOUR CODE HERE
    index_x = np.random.choice(img.shape[0], k, replace=False)
    index_y = np.random.choice(img.shape[1], k, replace=False)
    
    for i in range(k):
        current_cluster_centers[i][0] = img[index_x[i]][index_y[i]]


def kmeans(img, k):
    """Main k-means function iterating over max_iterations and stopping if
    the error rate of change is less then 2% for consecutive iterations, i.e. the
    algorithm converges. In our case the overall error might go up and down a little
    since there is no guarantee we find a global minimum.
    """
    print("Selected k is: {}\n".format(k))
    max_iter = 10
    max_change_rate = 0.02
    dist = sys.float_info.max
    h1,w1 = img.shape[:2]
    current_cluster_centers = np.zeros((k, 1, 3), np.float32)
    
    clustermask = np.zeros((h1, w1, 1), np.uint8)
    result = np.zeros((h1, w1, 3), np.uint8)
    
    initialize(img,k,current_cluster_centers)
    for i in range(max_iter):
        print("Iteration cycle: {}/{}".format(i+1,max_iter))
        overall_dist = assign_to_current_mean(img, result, clustermask, current_cluster_centers)
        print("Overall Error/Distance: {}".format(overall_dist))
        difference_in_dist = np.abs(overall_dist - dist)
        distance_avg = (overall_dist + dist) / 2
        diff_dist_percent = difference_in_dist / distance_avg
        print("Change rate is: {}".format(diff_dist_percent))
        if diff_dist_percent < max_change_rate:
            return result
        cluster_dict = {}
        update_mean(img, clustermask, current_cluster_centers, cluster_dict)
        dist = overall_dist
    
    # initializes each pixel to a cluster
    # iterate for a given number of iterations or if rate of change is
    # very small
    # YOUR CODE HERE

    return result

def get_images(img_location):
    imgraw = cv2.imread(img_location)
    scaling_factor = 0.5
    imgraw_scaled = cv2.resize(imgraw, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return imgraw, imgraw_scaled

# num of cluster
k = 4
img_raw, img_scaled = get_images("./data/Lenna.png")
result = kmeans(img_scaled,4)

result_lab = cv2.cvtColor(result, cv2.COLOR_BGR2Lab)
result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
result_luv = cv2.cvtColor(result, cv2.COLOR_BGR2Luv)
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

              
h1, w1 = result.shape[:2]
h2, w2 = img_scaled.shape[:2]
vis = np.zeros((max(h1, h2) * 2, w1 * 3, 3), np.uint8)
vis[:h1, :w1] = result
vis[:h2, w1:w1 + w2] = img_scaled
vis[h1:, :w1] = result_lab
vis[h2:, w1:w1 + w2] = result_hsv
vis[:h1, w1*2:w1*3] = result_luv
vis[h2:, w1*2:w1*3] = result_rgb
cv2.imshow("Color-based Segmentation Kmeans-Clustering", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()

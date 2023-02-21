#!/usr/bin/env python
# coding: utf-8

# In[2]:



import numpy as np
from skimage import io, img_as_float
import os

# k- means-clustering function
def clustering(image_vectors, k, num_iterations):
    labels = np.full((image_vectors.shape[0],), -1)
    clstr_proto = np.random.rand(k, 3)
    for i in range(num_iterations):
        points_label = [None for k_i in range(k)]
        for color_i, color in enumerate(image_vectors):           
            color_row = np.repeat(color, k).reshape(3, k).T          
            closest_label = np.argmin(np.linalg.norm(color_row - clstr_proto, axis=1))
            labels[color_i] = closest_label
            if (points_label[closest_label] is None):
                points_label[closest_label] = []
            points_label[closest_label].append(color)       
        for k_i in range(k):
            if (points_label[k_i] is not None):
                length = len(points_label[k_i])
                new_cluster_prototype = np.asarray(points_label[k_i]).sum(axis=0) / length
                clstr_proto[k_i] = new_cluster_prototype
    return (labels, clstr_proto)


imageFile ="/Users/nkitharamisetty/Desktop/Penguins.jpg"
image = io.imread(imageFile)[:, :, :3] 
image = img_as_float(image)
image_name = image
outputFile = "Penguins_k20.jpg"  
image_vectors = image.reshape(-1, image.shape[-1])
labels, color_centroids = clustering(image_vectors, k=20, num_iterations=10)
output_image = np.zeros(image_vectors.shape)
for i in range(output_image.shape[0]):
    output_image[i] = color_centroids[labels[i]]
image_dimensions = image.shape
output_image = output_image.reshape(image_dimensions)
io.imsave(outputFile , output_image)
print('Image Compression is Completed and Saved')
sizeBefore = os.stat(imageFile)
sizeAfter = os.stat(outputFile)
print("Image size before Compression: ",sizeBefore.st_size/1024,"KB")
print("Image size after Compression: ",sizeAfter.st_size/1024,"KB")


# In[ ]:





# In[ ]:





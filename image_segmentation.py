'''
Cálculo de areas en base a clusterización utilizando k-means, en imagenes satelite
parametros:
-- file imagen satelite
-- area area en km2 representada en la imagen satelite
se utilizan 3 clusters por imagen, (conversion a niveles de gris) con salto amplio
en la polarización de la imagen


MIT License
'''
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
import argparse
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=True,
                help="satellite image")
ap.add_argument("-a", "--area", required=True,
	help="satellite image area in km2")

args = vars(ap.parse_args())
# open file name
filename = args["file"]
area = args["area"]
I = Image.open(filename)

#resize and paint with axis
plt.figure(figsize=(8,8))
plt.imshow(I)
plt.axis('on')
plt.show()

# paint in gray scale
I1 = I.convert('L')
I2 = np.asarray(I1,dtype=np.float)

plt.figure(figsize=(8,8))
plt.imshow(I2,cmap='gray')
plt.axis('on')
plt.show()

# define 3 clusters and grey scale (black, white, mix scale)
X = I2.reshape((-1, 1))
k_means = KMeans(n_clusters=3)
'''
k_means=KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=3, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)
'''

k_means.fit(X)

centroides = k_means.cluster_centers_
etiquetas = k_means.labels_

I2_compressed = np.choose(etiquetas, centroides)
I2_compressed.shape = I2.shape

plt.figure(figsize=(8,8))
plt.imshow(I2_compressed,cmap='gray')
plt.axis('on')
plt.show()

I2 = (I2_compressed-np.min(I2_compressed))/(np.max(I2_compressed)-np.min(I2_compressed))*255
I2 = Image.fromarray(I2.astype(np.uint8))
w, h =I2.size
colors = I2.getcolors(w * h)
print('density points:',colors)


print (u'Área = ',  float(area)*float(colors[0][0])/float(w*h), 'km2')
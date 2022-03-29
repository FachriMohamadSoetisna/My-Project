# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 20:02:09 2021

@author: Yohanes Irfon & Fachri Mohamad Soetisna
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from sklearn.metrics.pairwise import euclidean_distances


#Membaca judul img
img_files = [doc for doc in os.listdir() if doc.endswith('.jpg')]
#Memasukkan seluruh file ke list
img = []
for File in img_files:
    img.append(cv2.imread(File, cv2.IMREAD_GRAYSCALE))
    
#membuat sift
sift = cv2.xfeatures2d.SIFT_create()

#membuat keypoint dan descriptors untuk setiap file
keypoints = []
descriptors = []
for File in img:
    #Mendeteksi keypoint dan descriptor dari tiap file image
    k, d = sift.detectAndCompute(File, None)
    #memasukkan keypoint ke k
    keypoints.append(k)
    #memasukkan descriptors ke d
    descriptors.append(d)

#menggabungkan seluruh keypoint dan descriptors
kp = []
dsc = []
for a in range(len(keypoints)):
    #current untuk menyimpan keypoint dan descriptor buku ke-a
    current_key = keypoints[a]
    current_desc = descriptors[a]
    for b in range(len(current_key)):
        #memasukkan keypoint dan descriptors ke list
        kp.append(current_key[b])
        dsc.append(current_desc[b])
        
#clustering dengan k-means
kmeans = KMeans(n_clusters=100, random_state=0).fit(dsc)

#simpan cluster label
resCluster = kmeans.labels_

#mengambil centroid tiap cluster
centroids  = kmeans.cluster_centers_

#menghitung intracluster distance dari setiap cluster
intra_dis = []
for i in range(100):
    current_cluster = []
    current_distance = []
    for j in range(len(resCluster)):
        if resCluster[j] == i:
            current_cluster.append(dsc[j])
    current_distance = euclidean_distances(current_cluster)
    avg_dis = current_distance.mean()
    intra_dis.append(avg_dis)
    
#membuat histogram untuk intracluster distance
plt.hist(intra_dis, bins = 50)

#menghitung jumlah anggota setiap cluster
cluster_member = []
for a in range(100):
    counter = 0
    for b in range(len(resCluster)):
        if resCluster[b] == a :
            counter = counter + 1
    cluster_member.append(counter)
    print(cluster_member)
    
#membuat histogram untuk jumlah anggota
plt.hist(cluster_member, bins = 10)

# menghitung upper bound
upper = []
centroidText = []
for i in range(len(intra_dis)):
    if intra_dis[i] <= 350 and cluster_member[i] >= 1000 :
        upper.append(i)
        centroidText.append(centroids[i])

# Cek pola untuk intracluster distance pada cluster yang sudah diaggap sebagai text
upper_dis = []
for c in range(len(upper)):
    for e in range(len(intra_dis)):
        if upper[c] == e:
            upper_dis.append(intra_dis[e])
            
#membuat histogram untuk melihat pola intracluster distance
plt.hist(upper_dis, bins = 20)

# klasifikasi untuk foto dataset baru
newImg = cv2.imread("newimg.jpg", cv2.IMREAD_GRAYSCALE)

# mendeteksi keypoint dan descriptor dari foto dataset baru
newKeypoints, newDescriptors = sift.detectAndCompute(newImg, None)

#normalisasi vector di newDescriptors dan menghitung jarak dari descriptor baru ke centroid yang memenuhi syarat cluster huruf
descriptorDist = []
for m in range(len(newDescriptors)):
    for n in range(len(centroidText)):
        dist = np.linalg.norm(newDescriptors[m] - centroidText[n])
        descriptorDist.append(dist)
        
#membuat histogram untuk melihat threshold huruf atau bukan
plt.hist(descriptorDist, bins = 10)

# Membagi descriptor untuk teks dan gambar
textKeypoint = []
imgKeypoint = []
for f in range(len(newDescriptors)):
    for h in range(len(centroidText)):
        dist = np.linalg.norm(newDescriptors[f] - centroidText[h])
        if dist < 300:
            textKeypoint.append(newKeypoints[f])
            break
    if dist >= 300:
        imgKeypoint.append(newKeypoints[f])
                
# membuat keypoint untuk cluster text dari foto dataset baru
newImg = cv2.drawKeypoints(newImg, textKeypoint, None, color = (0, 255, 0))
# membuat keypoint untuk cluster gambar dari foto dataset baru
newImg = cv2.drawKeypoints(newImg, imgKeypoint, None, color = (0, 0, 255))

cv2.imwrite("new.jpg", newImg)

# pie chart untuk keypoint huruf dan gambar
label = np.array(["Keypoint Huruf", "Keypoint Gambar"])
jumlahKeypoint = np.array([22167, 17254])
plt.pie(jumlahKeypoint, labels = label)
plt.show()

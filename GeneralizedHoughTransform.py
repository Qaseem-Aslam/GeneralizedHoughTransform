import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
print('row')
print('cols')
img = cv2.imread('face1.jpg',0)
img2 = cv2.imread('face.jpg',0)
#edgesmap = cv2.Canny(img,200,100)
edgesmap=img
[rows,cols]=edgesmap.shape
plt.imshow(edgesmap,cmap='gray')
R_table1=np.empty(180,dtype=np.object)
R_table2=np.empty(180,dtype=np.object)
for i in range(0,180):
    R_table1[i]=[]
    R_table2[i]=[]


sobelx = cv2.Sobel(edgesmap, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(edgesmap, cv2.CV_64F, 0, 1, ksize=5)
Edgex=[]
Edgey=[]
for x in range(0,rows):
    for y in range(0,cols):
        if(edgesmap[x][y]!=0):
            Edgex.append(x)
            Edgey.append(y)
avgx = round(sum(Edgex)/len(Edgex));
avgy = round(sum(Edgey)/len(Edgey));

if(avgx>rows):
    avgx=avgx%rows
if(avgy>cols):
    avgy=avgy%rows

for p in range(0,len(Edgex)):
   r = math.sqrt(pow((Edgex[p]-avgx),2)+pow((Edgey[p]-avgy),2))
   beta = math.atan2((Edgey[p]-avgy),(Edgex[p]-avgx))
   arah = math.atan2(sobely[Edgex[p]][Edgey[p]],sobelx[Edgex[p]][Edgey[p]]);
   arah = abs(arah)
   angle = round(arah)
   R_table1[angle].append(r)
   R_table2[angle].append(beta)
edges = cv2.Canny(img2,200,100)
[row,col]=edges.shape
sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=5)
Acc=np.zeros((row,col))
for x in range(0,row):
    for y in range(0,col):
        if(edges[x][y]!=0):
            fi = round(math.atan2(sobely[x][y],sobelx[x][y]))
            for beta in R_table2[fi]:
                r = R_table1[fi][i]
                beta = R_table2[fi][i]
                print(r)
                xc = abs(round(x+r*math.cos(beta)));
                yc = abs(round(y+r*math.sin(beta)));
                if(xc >= row):
                    xc = row-1
                if(yc >= col):
                    yc = col-1
                Acc[xc][yc] = Acc[xc][yc]+1

max=-9999999

for x in range(0,row):
    for y in range(0,col):
        if(Acc[x][y]>max):
            xco=x
            yco=y
            max=Acc[x][y]


print('center point at ('+str(xco)+','+str(yco)+')')
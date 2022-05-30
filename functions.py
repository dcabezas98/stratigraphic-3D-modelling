# -*- coding: utf-8 -*-

import numpy as np
import plotly.offline as go_offline
import plotly.graph_objects as go
import pandas as pd

import shapely.geometry as geometry

from shapely.geometry import Polygon
from scipy.spatial import ConvexHull

def coordinates(data,positions):
    p=positions
    dat=open(data,'r')
    lg=dat.readlines()
    n_line=len(lg)
    x=[]
    y=[]
    z=[]
    for i in range(1,n_line):
        split_line=lg[i].split(",")
        xyz_t=[]
        x.append(float(split_line[p[0]].rstrip()))
        y.append(float(split_line[p[1]].rstrip()))
        z.append(float(split_line[p[2]].rstrip()))
    return [x,y,z]

def bounds(list):
    x_min=min(list[0])
    x_max=max(list[0])
    y_min=min(list[1])
    y_max=max(list[1])
    w=x_max-x_min 
    h=y_max-y_min
    return [x_min,x_max,y_min,y_max,w,h]

def bounds_join(c1,c2):
    return[min(c1[0],c2[0]),
           max(c1[1],c2[1]),
           min(c1[2],c2[2]),
           max(c1[3],c2[3]),
           max(c1[1],c2[1])-min(c1[0],c2[0]),
           max(c1[3],c2[3])-min(c1[2],c2[2])
           ]

def nearby(xyz,polyg,dis):
    x=xyz[0]
    y=xyz[1]
    z=xyz[2]
    n=len(x)
    positions=[i for i in range(n) if polyg.distance(geometry.Point(x[i],y[i]))< dis]
    rx=[x[i] for i in positions]
    ry=[y[i] for i in positions]
    rz=[z[i] for i in positions]
    return [rx,ry,rz]


def data_p(list,names,colors,symbols,siz):
    n=len(list)
    return [go.Scatter3d(x=list[i][0], y=list[i][1], z=list[i][2],
            mode ='markers',
            name=names[i],
            marker = dict(size = siz,
                          color =colors[i],
                          opacity = 1,
                          symbol=symbols[i])
                        )
          for i in range(n) ]

def zipxyz(points):
    x=points[0]
    y=points[1]
    z=points[2]
    n=len(x)
    return np.array([[x[i],y[i],z[i]] for i in range(n)]) 

def dis(a,b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2*0.1)

def within(x,list,n):
    return [y for y in list if dis(x,y)<n]
    

def within2(list1,list2,n):
    if len(list1)==0:
        return list1
    elif len(list2)==0:
        return list2
    else:
        l=[within(x,list2,n) for x in list1]
        return np.unique(np.concatenate(l),axis=0)
    

def grouping(list1,list2,dist):
    newlist1=within2(list1,list2,dist)
    newlist2=within2(newlist1,list2,dist)
    if len(newlist1)==len(newlist2):
        return newlist1
    else:
        return grouping(newlist2,list2,dist)

def d_ss(vhull,shull,nombre,alpha,opc,col):
    return go.Mesh3d(x=vhull[:, 0],y=vhull[:, 1], z=vhull[:, 2], 
                        name=nombre,
                        showlegend=True,
                        colorbar_title=nombre,
                        color=col, 
                     i=shull[:, 0], j=shull[:, 1], k=shull[:, 2],
                        opacity=opc,
                        alphahull=alpha,
                        showscale=False
                       ) 

def data_lit(litosomes,names,alpha,opc,col):
    n=len(litosomes)
    return [d_ss(litosomes[i].points,litosomes[i].simplices,names[i],alpha,opc,col) for i in range(n)]

def data_lit_tr(litosomes,names,alpha,opc,col):
    n=len(litosomes)
    return [d_ss(litosomes[i].points,litosomes[i].vertices,names[i],alpha,opc,col) for i in range(n)]


def triangulation_edges(points, faces, nombre, linewidth=1.5):
    points = np.asarray(points)
    faces = np.asarray(faces)
    d = points.shape[-1]
    
    if d not in [2, 3] or faces.shape[-1] != 3:
        raise ValueError("your data are not associated to a 2d or 3d  triangulation\n\
                         points should be an array of ndim=2 or 3 and faces of ndim=3")
    
    tri_vertices = points[faces]
    Xe = []
    Ye = []
    if d == 3:
        Ze = []
    for T in tri_vertices:
        Xe += [T[k%3][0] for k in range(4)] + [None]
        Ye += [T[k%3][1] for k in range(4)] + [None]
        if d == 3: 
            Ze += [T[k%3][2] for k in range(4)] + [None] 
    if d == 2:
        return  go.Scatter(x=Xe,
                           y=Ye,
                           mode='lines',
                           legendgroup=grupo, 
                           name=nombre,
                           line_color ='rgb(50,50,50)', 
                           line_width=linewidth
                            )
    else:
        return go.Scatter3d(x=Xe,
                           y=Ye,
                           z=Ze,
                           mode='lines',
                           name=nombre,
                           line_color ='rgb(50,50,50)', 
                           line_width=linewidth) 

def data_tri(list,names):
    n=len(list)
    return [triangulation_edges(list[i].points,list[i].simplices,names[i],linewidth=1.5) for i in range(n)]

#DISTANCE FUNCTION
def distance(x1,y1,x2,y2):
    d=np.sqrt((x1-x2)**2+(y1-y2)**2)
    return d

#CREATING IDW FUNCTION
def idw_npoint(xz,yz,x,y,z,n_point,p):
    r=5 #block radius iteration distance
    nf=0
    while nf<=n_point: #will stop when np reaching at least n_point
        x_block=[]
        y_block=[]
        z_block=[]
        r +=10 # add 10 unit each iteration
        xr_min=xz-r
        xr_max=xz+r
        yr_min=yz-r
        yr_max=yz+r
        for i in range(len(x)):
            # condition to test if a point is within the block
            if ((x[i]>=xr_min and x[i]<=xr_max) and (y[i]>=yr_min and y[i]<=yr_max)):
                x_block.append(x[i])
                y_block.append(y[i])
                z_block.append(z[i])
        nf=len(x_block) #calculate number of point in the block
    
    #calculate weight based on distance and p value
    w_list=[]
    for j in range(len(x_block)):
        d=distance(xz,yz,x_block[j],y_block[j])
        if d>0:
            w=1/(d**p)
            w_list.append(w)
            z0=0
        else:
            w_list.append(0) #if meet this condition, it means d<=0, weight is set to 0
    
    #check if there is 0 in weight list
    w_check=0 in w_list
    if w_check==True:
        idx=w_list.index(0) # find index for weight=0
        z_idw=z_block[idx] # set the value to the current sample value
    else:
        wt=np.transpose(w_list)
        z_idw=np.dot(z_block,wt)/sum(w_list) # idw calculation using dot product
    return z_idw


def interpolation(list_of_points,n,bounds):
    [x_min,x_max,y_min,y_max,w,h]=bounds
    [x,y,z]=list_of_points
    wn=w/n #x interval
    hn=h/n #y interval
    #list to store interpolation point and elevation
    y_init=y_min
    x_init=x_min
    x_idw_list=[]
    y_idw_list=[]
    z_head=[]
    for i in range(n):
        xz=x_init+wn*i
        yz=y_init+hn*i
        y_idw_list.append(yz)
        x_idw_list.append(xz)
        z_idw_list=[]
        for j in range(n):
            xz=x_init+wn*j
            z_idw=idw_npoint(xz,yz,x,y,z,5,1.5) #min. point=5, p=1.5
            z_idw_list.append(z_idw)
        z_head.append(z_idw_list)
    return [z_head,x_idw_list,y_idw_list]

def cutting(list_of_points,polyg,dis):
    pc=list_of_points.copy()
    m=len(pc[1])
    for i in range(m):
        for j in range(m):
            if polyg.distance(geometry.Point(pc[1][i],pc[2][j]))>dis:
                pc[0][j][i]=np.nan
    return pc

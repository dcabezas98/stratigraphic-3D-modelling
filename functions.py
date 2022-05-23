# -*- coding: utf-8 -*-

import numpy as np
import plotly.offline as go_offline
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import shapely.geometry as geometry

from shapely.geometry import Polygon
from descartes import PolygonPatch
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

import os

from concave_hull import alpha_shape
from sklearn.neighbors import KNeighborsClassifier

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

def distance(a,b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def within(x,list,n):
    return [y for y in list if distance(x,y)<n]
    

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
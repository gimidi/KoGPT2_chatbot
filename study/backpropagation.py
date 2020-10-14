#!/usr/bin/env/python 3
import math
import random

a = random.uniform(-1,1)
b = random.uniform(-1,1)
c = random.uniform(-1,1)
d = random.uniform(-1,1)
e = random.uniform(-1,1)
f = random.uniform(-1,1)
g = random.uniform(-1,1)
h = random.uniform(-1,1)
i = random.uniform(-1,1)

eta = 0.0001

def sigmoid(x):
    return 1.0/(1.0+math.exp(-x))

data = [(5,10), (7,18), (10,8), (15,15), (20,3), (23,12)]
labels = [-1, -1, +1, +1, -1, -1]
for counter in range(1000):
    da = db = dc = dd = de = df = dg = dh = di = 0.0
    for idx in range(len(data)):
        x = float(data[idx][0])
        y = float(data[idx][1])
        z1 = a*x+b*y+c
        X = sigmoid(z1)
        z2 = d*x+e*y+f
        Y = sigmoid(z2)
        z = g*X+h*Y+i
        Z = labels[idx]

        dz = 2*(z-Z)
        dz1 = dz*g*sigmoid(z1)*(1.0-sigmoid(z1))
        dz2 = dz*h*sigmoid(z2)*(1.0-sigmoid(z2))
        da+=dz1*x
        db+=dz1*y
        dc+=dz1
        dd+=dz2*x
        de+=dz2*y
        df+=dz2
        dg+=dz*X
        dh+=dz*Y
        di+=dz

    a = a-eta*da
    b = b-eta*db
    c = c-eta*dc
    d = d-eta*dd
    e = e-eta*de
    f = f-eta*df
    g = g-eta*dg
    h = h-eta*dh
    i = i-eta*di

print("a=%lf, b=%lf, c=%lf, d=%lf, e=%lf, f=%lf, g=%lf, h=%lf,i=%lf"%(a,b,c,d,e,f,g,h,i))

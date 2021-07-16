
import pandas as pd
import matplotlib.pyplot as plt 
import time
import datetime
import numpy as np
import time

import numba
from numba import jit

def temps(x):
    return datetime.datetime.strptime(x,  "%m/%d/%Y")

df = pd.read_csv('data.csv')
df2 = pd.read_excel('Bloom_EUR_OIS.xlsx')
df2['Payment Date'] = df2['Payment Date'].apply(temps)
#df.columns=['t','DF(t)']

df['t'] = pd.to_datetime(df['t'])
df['t'] = df['t'].apply(datetime.datetime.timestamp)
df['t'] = df['t']/86400
df['V'] = np.cumsum(df['MarketRate']*df['DF(t)'])

n = len(df)
N = 100
R = .4
LGD = 1-R


@jit(nopython = True)
def delta(t1,t2):
    return (t2-t1)#.days

@jit(nopython = True)      
def P(t,T):
    return np.average(df[df.t <= T]['DF(t)'])

@jit(nopython = True)
def F(t,S,T):
    return (P(t,S)/P(t,T)-1)/delta(S,T)
    
@jit(nopython = True)  
def m_rate(date):
    k=0
    max_k = len(df2.T)
    while not df2.T[k]['Payment Date']<= date :#<= df2.T[k+1]['Payment Date']:
        k+=1
        if k == max_k-2:
            print("error")
            break

    return df2.T[k]['Market Rate']

@jit(nopython = True) 
def append_m_rate():
    df['MarketRate'] = np.vectorize(m_rate)(df['t'])



    
@jit(nopython = True)
def polynome(t, p = .5):
    
    #t = df['t'][i]
    s = 0
    
    for j in range(1+list(df['t']).index(t),n):
        T    = df['t'][j]
        T_1  = df['t'][j-1]
        
        s +=    P(t,T)*         \
                p**delta(t,T)*  \
                F(t,T_1,T)*     \
                delta(T_1,T)    \
                +                    \
                (p**delta(t,T_1) - p**delta(t,T))*(1-LGD)


    s += P(t, list(df['t'])[-1])* p**delta(t, list(df['t'])[-1])

    return s



@jit(nopython = True)
def solve_polynome(t,V, epsilon = 1e-3):
    a = 0
    b = 1
    p = (b+a) / 2

    while np.abs(polynome(t,p)+V/N)>epsilon:
        if polynome(t,p) < 0:
            a=p
        else:
            b=p
        p=(b+a)/2

    return p 

    



print("évaluation du polynome pour différentes valeurs de t \navec p fixé p = 0.5 \n\n")

for i in range(len(df['t'])):
    
    t_i = df['t'][i]
    V_i = df['V'][i]
    
    print("==================")
    t1 = time.time()
    print("p = {}".format(solve_polynome(t_i, V_i)))
    print("Execution time = {}s".format(round(time.time()-t1,3)))
    #stripper.polynome(t)
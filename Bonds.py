import pandas as pd
import matplotlib.pyplot as plt 
import time

import numpy as np
import scipy as sp
import scipy.interpolate

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Choose market data",)
    parser.add_argument("--start",    default='2020-12-31', help="Start date    | format : 'yyyy-mm-dd")
    parser.add_argument("--maturity", default='2071-01-05', help="Maturity date | format : 'yyyy-mm-dd")
    parser.add_argument("--price", default=1.04, help="Price",type=float)
    parser.add_argument("--frequence", default=3,help="Frequence (en mois)",type=int)
    parser.add_argument("--taux", default = 4/100, help="Taux fixe",type=float)
    parser.add_argument("--plotting",action="store_true", help="Plot or not")

    args = parser.parse_args()

    return args

def get_month(x):
    return x.month

def get_day(x):
    return x.day

def log_interp1d(xx, yy, kind='linear'):

    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind)
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp



class Bond():
    def __init__(self, start='2020-12-31', end='2071-01-05', price= 1.04, freq = 3, K = 4/100): #yyyy-mm-dd
        

            
        self.price = price
        self.freq = freq
        self.K = K
    
        self.start = datetime.strptime(start,'%Y-%m-%d')
        try:
            self.end = datetime.strptime(end,'%Y-%m-%d')
        except:
            if type(end) is not str:
                self.end = end
            else:
                self.end = datetime.strptime(end,'%m/%d/%Y')


        


        
        
    def F(self,S,T):
        return (self.DF[S]/self.DF[T]-1) / (self.freq*self.delta)

    

    def polynome(self, p,LGD=.6):

        n = len(self.DF)
        zero_target = self.DF[n-1]*p**self.BC[n-1] 
        
        for i in range(len(self.DF)):
            zero_target += self.DF[i]* \
                            (
                            p**self.BC[i] * self.F(i-1,i) * self.delta + \
                            (p**self.BC[i-1] - p**self.BC[i] )*(1-LGD)
                            )
            
            
        zero_target += self.DF[n-1] * p**self.BC[n-1]
        zero_target -= self.price
        
        return zero_target

    
    
    def solve_polynome(self, epsilon = 1e-8):

        left = 0
        right= 1

        while right-left>epsilon :

            mid = (left+right) / 2
            if (self.polynome(left))*(self.polynome(mid)) < 0:
                right = mid
            else:
                left = mid

        return left
    
    
    def get_PS_1year(self):
        self.p = self.solve_polynome()
        return self.p
    
    def get_PS(self):
        
        self.PS_1year = self.get_PS_1year()
        
        self.PS = [self.PS_1year**delta for delta in self.BC]
        
        self.df['PS'] = self.PS
        self.df['PD'] = 1 - self.df['PS']
        
    def get_PD(self):
        self.get_PS()


    def stripping(self, data):

        if type(data) == str:
            self.data = pd.read_excel(data)
        else:
            self.data = data
        


        
        self.total_dates = [datetime.strptime(self.data['Payment Date'][0],'%m/%d/%Y')]
        
        while self.total_dates[-1] < self.end:
            self.total_dates.append(self.total_dates[-1] + relativedelta(months=+self.freq))
            
        while self.total_dates[-1] != self.end: #TODO change that
            self.total_dates[-1] -= timedelta(days=1)
            
        self.total_days = [(date-self.start).days for date in self.total_dates]
        


        data_t  = [(datetime.strptime(date,'%m/%d/%Y') - self.start).days for date in self.data['Payment Date']]
        data_df = self.data['Discount']
    
        self.DF = np.vectorize(log_interp1d(data_t, data_df))(self.total_days)



        self.delta = 1/365
        self.BC = np.array([day * self.delta for day in (self.total_days)])
        self.df = pd.DataFrame({'Maturity':self.total_dates,'Days':self.total_days,'DF':self.DF,'BC':self.BC})
        
        self.CF = list(self.K * self.df['DF'] * self.df['BC'])
        self.CF[-1] += 1
        
        self.df['CF'] = self.CF


        return 1 - self.get_PS_1year()

        





if __name__ == '__main__':

    print("=========================")
    t1 = time.time()
    args = main()
    print("Argument parsing time = {}s".format(round(time.time()-t1,3)))

    print("=========================")
    t2 = time.time()
    stripper = Bond(args.data, args.start, args.maturity, args.price, args.frequence, args.taux)
    print("Initialisation time = {}s".format(round(time.time()-t2,3)))

    print("=========================")
    t3 = time.time()
    stripper.get_PS()
    print("Stripping time = {}s".format(round(time.time()-t3,3)))


    stripper.df[['Maturity','PD']].to_csv('PD.csv')

    if args.plotting:
        plt.plot(stripper.df['Maturity'],stripper.df['PD'])
        plt.xlabel('Maturity Date')
        plt.ylabel('PD')
        plt.title('PD(T)')
        plt.show()
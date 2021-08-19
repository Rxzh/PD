import pandas as pd
import matplotlib.pyplot as plt 
import time

import numpy as np
import scipy as sp
import scipy.interpolate
from scipy.optimize import minimize

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Choose market data",)
    parser.add_argument("--start",    default='2020-12-31', help="Start date    | format : 'yyyy-mm-dd")
    parser.add_argument("--maturity", default='2071-01-05', help="Maturity date | format : 'yyyy-mm-dd")
    parser.add_argument("--price", default=1.04, help="Price",type=float)
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

def log_interp1d_neg(xx, yy, kind='linear'):

    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind)
    log_interp = lambda zz: -np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp



class Bond():
    def __init__(self, start='2020-12-31', end='2071-01-05', price= 1.04, K = 4/100,coupon=10): #yyyy-mm-dd
        
        self.coupon = coupon
        self.price = price
        self.K = K
        self.LGD = .6
        self.recovery_rate = 1 - self.LGD
        self.principal_payment = 100
    
        self.start = datetime.strptime(start,'%Y-%m-%d')
        try:
            self.end = datetime.strptime(end,'%Y-%m-%d')
        except:
            if type(end) is not str:
                self.end = end
            else:
                self.end = datetime.strptime(end,'%m/%d/%Y')
        
        self.maturity_years = (self.end - self.start).days / 365


    

    def polynome(self,P):
        #Need = self.times and self.cashflow, self.principal_payment, self.risk_adjusted_discount_rate, self.recovery_rate
            x_prob_default_exp = 0

            for i in range(len(self.times)):

                #if there is only one payment remaining
                if len(self.times) == 1:
                    x_prob_default_exp += ((self.cashflows[i]*(1-P) + self.cashflows[i]*self.recovery_rate*P) \
                                        #/ np.power((1 + self.risk_adjusted_discount_rate), self.times[i]))
                                        * self.DF[i]) # ou self.DF[self.times[i]]

                #if there are multiple payments remaining
                else:

                    if self.times[i] == 1:
                        x_prob_default_exp += ((self.cashflows[i]*(1-P) + self.principal_payment*self.recovery_rate*P) \
                                                #/np.power((1 + self.risk_adjusted_discount_rate), self.times[i]))
                                                * self.DF[i]) # ou self.DF[self.times[i]]


                    else:
                        x_prob_default_exp += ((np.power((1-P), self.times[i-1])*(self.cashflows[i]*(1-P) + self.principal_payment*self.recovery_rate*P)) \
                                                #/ np.power((1 + self.risk_adjusted_discount_rate), self.times[i])
                                                * self.DF[i])
            

            return (x_prob_default_exp - self.price)**2

    
    def probability_of_default(self):

        self.times = np.arange(1, self.maturity_years+1) 
        
        self.annual_coupon = self.coupon        
        
        # Calculation of Expected Cash Flow
        self.cashflows = np.array([])

        for _ in self.times[:-1]:
                self.cashflows = np.append(self.cashflows, self.annual_coupon)
        self.cashflows = np.append(self.cashflows, self.annual_coupon+self.principal_payment)
        

        implied_prob_default = minimize(self.polynome, x0=np.array([.5]),method='Powell')

        pd = max(0.0,min(implied_prob_default.x,1.0))

        return pd

    


    def stripping(self, data):

        if type(data) == str:
            self.data = pd.read_excel(data)
        else:
            self.data = data
        
        
        self.total_dates = [datetime.strptime(self.data['Payment Date'][0],'%m/%d/%Y')]
        
        while self.total_dates[-1] < self.end:
            self.total_dates.append(self.total_dates[-1] + relativedelta(years=+1))
            
        while self.total_dates[-1] != self.end: #TODO change that
            self.total_dates[-1] -= timedelta(days=1)
            
        self.total_days = [(date-self.start).days for date in self.total_dates]
        self.total_years = np.array(self.total_days)/365
        


        data_t  = [(datetime.strptime(date,'%m/%d/%Y') - self.start).days for date in self.data['Payment Date']]
        data_df = self.data['Discount']
    
        self.DF = np.vectorize(log_interp1d(data_t, data_df))(self.total_days)


        self.delta = 1/365
 
        self.df = pd.DataFrame({'Maturity':self.total_dates,'Days':self.total_days,'DF':self.DF})
    
        self.PD_1y = self.probability_of_default()
  
        return self.PD_1y

    
    def reprice(self, PD_1y = None):

        if PD_1y is None:
            PD_1y = self.PD_1y

        price = 0.0
        for i in range(len(self.times)):

            #if there is only one payment remaining
            if len(self.times) == 1:
                price += ((self.cashflows[i]*(1-PD_1y) + self.cashflows[i]*self.recovery_rate*PD_1y) \
                            #/ np.power((1 + self.risk_adjusted_discount_rate), self.times[i]))
                            * self.DF[i]) # ou self.DF[self.times[i]]

                

            #if there are multiple payments remaining
            else:

                if self.times[i] == 1:
                    price += ((self.cashflows[i]*(1-PD_1y) + self.principal_payment*self.recovery_rate*PD_1y) \
                                            #/np.power((1 + self.risk_adjusted_discount_rate), self.times[i]))
                                            * self.DF[i]) # ou self.DF[self.times[i]]


                else:
                    price += ((np.power((1-PD_1y), self.times[i-1])*(self.cashflows[i]*(1-PD_1y) + self.principal_payment*self.recovery_rate*PD_1y)) \
                                            #/ np.power((1 + self.risk_adjusted_discount_rate), self.times[i])
                                            * self.DF[i])
        
        return price

        





        """
        price = 0.0
        n_remaining_payments = int(self.maturity_years) + 1
        for i in range(n_remaining_payments):

            ajout = self.PD_1y*(1-self.LGD) + (1-self.PD_1y) * self.annual_coupon/100
            ajout *= (1-self.PD_1y)**i
            ajout *= list(self.df['DF'])[i]

            price += ajout
        """

        #verifier que les coupons soient actualisés
        #actualiser le +100

        return 100*(price +(list(self.df['DF'])[-1] ))








if __name__ == '__main__':

    print("=========================")
    t1 = time.time()
    args = main()
    print("Argument parsing time = {}s".format(round(time.time()-t1,3)))

    print("=========================")
    t2 = time.time()
    stripper = Bond(args.data, args.start, args.maturity, args.price, args.taux)
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
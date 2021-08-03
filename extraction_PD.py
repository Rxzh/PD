import pandas as pd
import numpy as np
from time import time
from Bonds import Bond
import argparse
import scipy as sp
from datetime import datetime

def log_interp1d_neg(xx, yy, kind='linear'):

    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind)
    log_interp = lambda zz: -np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp


def generate_discount_rate_func(start):
    d_factor = pd.read_excel("DF(t) .xlsx")
    d_factor.columns=['date','df']

    def foo(x):
        return ((x-datetime.strptime(start,'%Y-%m-%d')).days)/365

    d_factor['date'] = d_factor['date'].apply(foo)

    z = 1/d_factor['date']*np.log(d_factor['df'])


    discount_rate_func = log_interp1d_neg(d_factor['date'], z,)
    return discount_rate_func



def pipeline(end, price,K,coupon,discount_rate_func, start='2021-07-29'):
    try:
        bond = Bond(start, end, price, K,coupon,discount_rate_func)
        return bond.stripping("Bloom_OIS_29072021.xlsx")
    except:
        return np.nan

def main(file_name):

    discount_rate_func = generate_discount_rate_func(start='2021-07-29')

    #df = pd.read_excel(file_name)
    df = pd.read_csv(file_name)

    df = df[df['Maturity'].notna()]

    df = df.iloc[: , 1:]
    df['Mid Price'].replace(',','',inplace=True)
    mid_prices = []
    DR_func = []

    for x in df['Mid Price']:
        if type(x) == str:
            mid_prices.append(x.replace(',',''))
        else:
            mid_prices.append(x)

        DR_func.append(discount_rate_func)

    df['Mid Price'] = mid_prices

    df['PD_1y'] = np.vectorize(pipeline)(end=df['Maturity'],
                                         price=df['Mid Price'].astype(float),
                                         K=df['Cpn'].astype(float)/100, 
                                         coupon=df['Cpn'], 
                                         discount_rate_func=DR_func)

    
 
    df['PD_1y'].to_csv('PD_1y.csv')

    df.to_csv('market_data_pd.csv')




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--bonds", help="Choose the bonds data excel name")
    parser.add_argument("--verbose",action="store_true", help="Increase Verbosity")
    args = parser.parse_args()

    t1 = time()
    main(args.bonds)
    print("Total execution time = {}s".format(round(time()-t1,3)))
    


import pandas as pd
import numpy as np
from time import time
from Bonds import Bond
import argparse
import scipy as sp
from datetime import datetime

import matplotlib.pyplot as plt



def pipeline(end, price,K,coupon, start='2021-07-29'):

    try:
        bond = Bond(start, end, price, K,coupon)
        pd = bond.stripping("Bloom_OIS_29072021.xlsx")
        reprice = bond.reprice()

        return pd, price, reprice
    except: 
        return -1,-1,-1




def main(file_name):

    #df = pd.read_excel(file_name)
    df = pd.read_csv(file_name)

    df = df[df['Maturity'].notna()]

    df = df.iloc[: , 1:]
    df['Mid Price'].replace(',','',inplace=True)
    mid_prices = []


    for x in df['Mid Price']:
        if type(x) == str:
            mid_prices.append(x.replace(',',''))
        else:
            mid_prices.append(x)



    df['Mid Price'] = mid_prices

    df['PD_1y'],_,df['reprice'] = np.vectorize(pipeline)(end=df['Maturity'],
                                         price=df['Mid Price'].astype(float),
                                         K=df['Cpn'].astype(float)/100, 
                                         coupon=df['Cpn'])

    
    df['Error %'] = 100*np.abs((df['Mid Price'] - df['reprice'])/df['Mid Price'])
 
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
    


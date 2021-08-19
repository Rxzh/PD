import pandas as pd
import numpy as np
from time import time,sleep
from Bonds import Bond
import argparse
import scipy as sp
from datetime import datetime
from threading import Thread
import matplotlib.pyplot as plt


class pipeline_thread:
    def __init__(self, end, price,K,coupon, start='2021-08-11'):
        self.price = price
        self.bond = Bond(start, end, price, K,coupon)
    
    def stripping(self):
        try:
            self.pd = self.bond.stripping("Bloom_OIS_29072021.xlsx")
            self.reprice = self.bond.reprice()
            self.maturity_years = self.bond.maturity_years
            return self.pd, self.price, self.reprice, self.maturity_years
        except: 
            return -1,-1,-1,-1

def pipeline(end, price,K,coupon, start='2021-08-11'):


    try:
        bond = Bond(start, end, price, K,coupon)
        pd = bond.stripping("Bloom_OIS_29072021.xlsx")
        reprice = bond.reprice()
        maturity_years = bond.maturity_years



        return pd, price, reprice, maturity_years
    except: 
        return -1,-1,-1,-1



def main_thread(file_name):
    df = pd.read_csv(file_name, sep=',')
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

    BONDS = [pipeline_thread(end=(df['Maturity'])[i], price=(df['Mid Price'].astype(float))[i],K=(df['Cpn'].astype(float)/100)[i],coupon=(df['Cpn'])[i]) for i in range(len(df['Maturity']))]

    THREADS = [Thread(target=bond.stripping) for bond in BONDS]

    for t in THREADS:
        t.start()


    df['PD_1y'] = [bond.pd for bond in BONDS]
    df['reprice'] = [bond.reprice for bond in BONDS]
    df['maturity_years'] = [bond.maturity_years for bond in BONDS]
    

    
    df['Error %'] = 100*np.abs((df['Mid Price'] - df['reprice'])/df['Mid Price'])
 
    df['PD_1y'].to_csv('PD_1y.csv')


    df.to_csv('market_data_pd.csv')




def main(file_name):

    #df = pd.read_excel(file_name)
    df = pd.read_csv(file_name, sep=',')

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

    sleep(2)

    df['PD_1y'],_,df['reprice'],df['maturity_years'] = np.vectorize(pipeline)(  end=df['Maturity'],
                                                                                price=df['Mid Price'].astype(float),
                                                                                K=df['Cpn'].astype(float)/100, 
                                                                                coupon=df['Cpn'])

    
    df['Error %'] = 100*np.abs((df['Mid Price'] - df['reprice'])/df['Mid Price'])
 
    df['PD_1y'].to_csv('PD_1y.csv')


    df.to_csv('market_data_pd.csv')




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--bonds", help="Choose the bonds data excel name")
    parser.add_argument("--method",default='vector', help="Choose the probability calculation method")
    parser.add_argument("--verbose",action="store_true", help="Increase Verbosity")
    args = parser.parse_args()

    t1 = time()

    if args.verbose:
        print("stripping PDS with method : " + args.method)
        
    if args.method == 'vector':
        main(args.bonds)
    
    elif args.method == 'threading':
        main_thread(args.bonds)

    print("Total execution time = {}s".format(round(time()-t1,3)))
    


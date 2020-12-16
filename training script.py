# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:55:35 2020

@author: ShinFuChen
"""

import numpy as np
import pandas as pd
import os
import statistics as stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.externals import joblib

# Setup

np.set_printoptions(suppress=True) #以小數點顯示
os.getcwd()




# Data preprocessing
#將'????'資料設為遺失值
file = 'CDA_data.csv'
data = pd.read_csv(open(file), header=0) 
data.where(data != '????', np.nan, inplace=True)
data.to_csv(file,index = False)#輸出檔案
#重新讀取資料
data = pd.read_csv(open(file), header=0) 
data['Date'] = pd.to_datetime(data['Date'],format='%Y/%m/%d %H:%M') #時間格式轉換
## 時間處理(參考智一)
#判斷時間間格
x = []
for i in range(1,len(data)):
    x = np.append(x,(data.loc[i,'Date'] - data.loc[i-1,'Date']).seconds)

timedelta = int(stats.mode(x)) #單位:sec
#補齊缺少的時間
len_t_NA = 0
if len(np.unique(x))>1: #若時間間格值>1種，則表示時間資料有缺
    #找出缺少資料的時間
    t_all = pd.date_range(data.loc[0,'Date'],data.loc[len(data)-1,'Date'], freq= str(timedelta) +'s')
    t_NA = pd.DataFrame(data = list(set(t_all).difference(set(data['Date']))), columns = ['Date'])
    len_t_NA = len(t_NA)
    col_NA = pd.DataFrame(columns = data.columns[1:],index = range(len_t_NA))
    data_NA = pd.concat([t_NA, col_NA], axis = 1)
    data = pd.concat([data, data_NA])
    data = data.sort_values(by='Date').reset_index(drop=True) #依時間重新排序
#
    
#先進行訓練
    
# 列出所有機台型號以方便判斷機台
df_ls=[]
#for i in range(1,16):
#    n = "CDA"+ str(i).zfill(2)
#    df_ls.append(n)
#empty1="CDA"+ str(8).zfill(2)
#empty2="CDA"+ str(9).zfill(2)
#
#df_ls.remove(empty1)
#df_ls.remove(empty2)
n_list=["01","02","03","04","05","06","07","10","11","12","13","14","15"]
#創建std_ls,用來存放標準差std
std_ls=[]
#先分出csv檔並運算模型
threshold=100
for n in n_list:
        ## 將資料集的編號指定出來 
        ## 排除開關機資料 用電流MAT
        avoid_df= data.filter(regex=("Date|"+ n)).query('CDA'+n+'_MAT'+"> @threshold")
        ## 按照y的數量去做
        ##2個y_H 01 02 03 04
        if int(n)< 5:
            X=avoid_df[['CDA'+n+'_MAT','CDA'+n+'_FIT','CDA'+n+'_IOT','CDA'+n+'_OMT','CDA'+n+'_IMT','CDA'+n+'_IAT','CDA'+n+'_IVT','CDA'+n+'_SPT']] #確認 X
            y1= avoid_df[['CDA'+n+'_LVT']]#確認Y
            y2= avoid_df[['CDA'+n+'_HVT']]#確認Y
            std1= avoid_df[['CDA'+n+'_LVT']].std().get('CDA'+n+'_LVT')#計算std
            std2= avoid_df[['CDA'+n+'_HVT']].std().get('CDA'+n+'_HVT')#計算std
            #set model
            poly_reg = PolynomialFeatures(degree=2)
            X1_poly = poly_reg.fit_transform(X)
            pr_model_LVT = LinearRegression()
            pr_model_HVT = LinearRegression()
            #fit y1 and yw
            pr_model_LVT.fit(X1_poly, y1)
            pr_model_HVT.fit(X1_poly,y2)
            #export model
            joblib.dump(pr_model_LVT,'CDA'+n+'_pr_model_LVT.pkl')
            joblib.dump(pr_model_HVT,'CDA'+n+'_pr_model_HVT.pkl')
            #append std
            std_ls.append(std1)
            std_ls.append(std2)
            #append df_ls
            df_ls.append('CDA'+n+'_LVT')
            df_ls.append('CDA'+n+'_HVT')
        ##3個y_H 05 06 07
        elif int(n)> 4 and int(n)< 8 :
            X=avoid_df[['CDA'+n+'_MAT','CDA'+n+'_SPT','CDA'+n+'_FIT','CDA'+n+'_IAT','CDA'+n+'_IOT','CDA'+n+'_OMT','CDA'+n+'_IMT','CDA'+n+'_SMT1','CDA'+n+'_SMT2','CDA'+n+'_SMT3']] #確認 X
            y1= avoid_df[['CDA'+n+'_LVT_1']]#確認Y
            y2= avoid_df[['CDA'+n+'_LVT_2']]#確認Y
            y3= avoid_df[['CDA'+n+'_HVT']]#確認Y
            std1= avoid_df[['CDA'+n+'_LVT_1']].std().get('CDA'+n+'_LVT_1')#計算std
            std2= avoid_df[['CDA'+n+'_LVT_2']].std().get('CDA'+n+'_LVT_2')#計算std
            std3= avoid_df[['CDA'+n+'_HVT']].std().get('CDA'+n+'_HVT')#計算stdY
            #set model
            poly_reg = PolynomialFeatures(degree=2)
            X1_poly = poly_reg.fit_transform(X)
            pr_model_LVT_1 = LinearRegression()
            pr_model_LVT_2 = LinearRegression()
            pr_model_HVT = LinearRegression()
            #fit y1 and yw
            pr_model_LVT_1.fit(X1_poly, y1)
            pr_model_LVT_2.fit(X1_poly, y2)
            pr_model_HVT.fit(X1_poly,y3)
            #export model
            joblib.dump(pr_model_LVT_1,'CDA'+n+'_pr_model_LVT_1.pkl')
            joblib.dump(pr_model_LVT_2,'CDA'+n+'_pr_model_LVT_2.pkl')
            joblib.dump(pr_model_HVT,'CDA'+n+'_pr_model_HVT.pkl')
            #append std
            std_ls.append(std1)
            std_ls.append(std2)
            std_ls.append(std3)
            #
            df_ls.append('CDA'+n+'_LVT_1')
            df_ls.append('CDA'+n+'_LVT_2')
            df_ls.append('CDA'+n+'_HVT')
        ##1個y_L 11 12 13 14
        elif int(n) > 10 and int(n)< 15 :
            
            X=avoid_df[['CDA'+n+'_MAT','CDA'+n+'_SMT','CDA'+n+'_OMT','CDA'+n+'_IMT','CDA'+n+'_IOT']] #確認 X
            y1= avoid_df[['CDA'+n+'_LVT']]#確認Y
            std1= avoid_df[['CDA'+n+'_LVT']].std().get('CDA'+n+'_LVT')#計算std
            #set model
            poly_reg = PolynomialFeatures(degree=2)
            X1_poly = poly_reg.fit_transform(X)
            pr_model_LVT = LinearRegression()
            pr_model_LVT.fit(X1_poly, y1)
            #export model
            joblib.dump(pr_model_LVT,'CDA'+n+'_pr_model_LVT.pkl')
            #append std
            std_ls.append(std1)
            df_ls.append('CDA'+n+'_LVT')
            
        ##2個y_L  10 15
        else:
            
            X=avoid_df[['CDA'+n+'_MAT','CDA'+n+'_SMT','CDA'+n+'_OMT','CDA'+n+'_IMT','CDA'+n+'_IOT','CDA'+n+'_OPT','CDA'+n+'_FIT','CDA'+n+'_SMT']] #確認 X
            y1= avoid_df[['CDA'+n+'_LVT']]#確認Y
            y2= avoid_df[['CDA'+n+'_HVT']]#確認Y
            std1= avoid_df[['CDA'+n+'_LVT']].std().get('CDA'+n+'_LVT')#計算std
            std2= avoid_df[['CDA'+n+'_HVT']].std().get('CDA'+n+'_HVT')#計算std
            #set model
            poly_reg = PolynomialFeatures(degree=2)
            X1_poly = poly_reg.fit_transform(X)
            pr_model_LVT = LinearRegression()
            pr_model_HVT = LinearRegression()
            #fit y1 and yw
            pr_model_LVT.fit(X1_poly, y1)
            pr_model_HVT.fit(X1_poly,y2)
            #export model
            joblib.dump(pr_model_LVT,'CDA'+n+'_pr_model_LVT.pkl')
            joblib.dump(pr_model_HVT,'CDA'+n+'_pr_model_HVT.pkl')
            #append std
            std_ls.append(std1)
            std_ls.append(std2)
            #
            df_ls.append('CDA'+n+'_LVT')
            df_ls.append('CDA'+n+'_HVT')
        
        
# 輸出 index of std
CDA_std = pd.DataFrame({'CDA':df_ls,'std':std_ls})
file='CDA_std.csv'
CDA_std.to_csv(file,index = False)

       
            
    
    

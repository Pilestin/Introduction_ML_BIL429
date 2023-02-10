import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split


data_frame = pd.read_csv("kc_house_data.csv")
data_frame = data_frame.drop(['id', 'date', 'yr_renovated','zipcode','lat','long'], axis=1)
isimler = []

for i in data_frame.columns:
    isimler.append(i)

isimler
data_frame = data_frame[isimler]


scaler = MinMaxScaler()
data_frame = pd.DataFrame(scaler.fit_transform(data_frame.values), columns=data_frame.columns)

X = data_frame.drop(["price"], axis=1)
Y = data_frame[["price"]]


x_train, x_test, y_train, y_test =  train_test_split(X,Y,train_size=0.7) # %70 ini test için ayır.


def Hq(X , Q_all:list):
    hq = 0 
    # Önce satırı al dizi olarak 
    # Satırı enumerate ile (indis,veri) olarak al
    # Q[indis] * veri
    for j, attr in enumerate(X, start=1): 
        hq = hq +  Q_all[j]*attr
    hq += Q_all[0]  
    return hq

def J_2(X, Y, Q_all:list ):
    """ VERİLEN Q DEĞERLERİ İÇİN TÜM TABLOYU DENER VE GERÇEK Y İLE FARKIN KARELERİNİ TOPLAR. BÖYLECE MALİYETİ ÖLÇER """

    # hx = Q0 + Q1*x1 + Q2x2  + . . .  
    
    total = 0
    satir_sayisi = len(X)
    kolon_sayisi = X.shape[1]
    result = 0 
    # hq = 0

    for i, satir in enumerate(X):
        # -------- bu kısım yukarıda Hq metoduna devredilmiştir -------------------
        # hq = 0
        # for j,attr in enumerate(satir , start=1):
        #     hq = hq +  Q[j]*attr
        # hq += Q[0] 
        # -------------------------------------------------------------------------
        # 
        # her satırı dizi olarak alıyorum ve Hq metoduna verip hipotezi uyguluyorum.
        
        total = total + ( Y[i][0] -  Hq(X=satir,Q_all = Q_all)  )**2
         

    result = (1/2*satir_sayisi)*total
    return result

# J_2(X=x_train , Y=y_train.values , Q_all=Q_all)

def J_derivate_2(X, Y, Q_all:list , k:int ):
    """GRADİENT D. ALGORİTMASI İÇİN MALİYETİN, TÜREVİNİN ALINDIKTAN SONRA, HESAPLANDIĞI KISIM"""
     
    total = 0
    result = 0
    satir_sayisi = len(X)
    if(k==0):
        for i, satir in enumerate(X):
            total = total + ( Hq(X=satir,Q_all = Q_all) - Y[i][0] )
    else:
        for i, satir in enumerate(X): 
            total = total + (Hq(X=satir,Q_all = Q_all) - Y[i][0]   )*X[i][k-1]

    result = (1/satir_sayisi) * total 
    return result 

q_temp = []  # Q listesinin değişimini iterasyon sonuna kadar tutacağımız geçici liste
Q_all  = []
# başlangıç değeri için kolon sayısı + 1 adet 1 ekleyeceğim
for i in range(len(X.columns)+1):
    Q_all.append(1)
    q_temp.append(1)

print(Q_all)

import copy

cost_all = []

def Gd_with_all(X , Y , Q_all:list, alfa=0.04, epoch= 100 ):
    
    satir_sayisi = len(X)
    kolon_sayisi = X.shape[1]
    
    # epoch sayısı kadar eğit
    for i in range(epoch):
        # her iterasyonda maliyet ? 
        temp_cost = J_2(X=x_test.values, Y=y_test.values, Q_all=Q_all)
        cost_all.append(temp_cost) 

        # Eğitim için tüm satırları gez. 
        for j in range(kolon_sayisi+1):
            q_temp[j] = Q_all[j] - alfa * J_derivate_2(X=X, Y=Y, Q_all=Q_all, k=j)   
        

        Q_all = copy.deepcopy(q_temp)

        if i % 50 == 0:
            print(f"{i}. Maliyetim = ",temp_cost )
            plt.scatter(x = range(len(cost_all)) ,y = cost_all , color="red")       
            # tanım
            plt.xlabel("epoch")
            plt.ylabel("cost")
            plt.title("Epoch - Cost")
            plt.show()    
    

    print(Q_all) 
    

Gd_with_all(X=x_train.values , Y=y_train.values, Q_all=Q_all, alfa=0.20 , epoch=200)
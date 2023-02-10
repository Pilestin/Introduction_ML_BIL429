import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
import copy

data_frame = pd.read_csv("kc_house_data.csv")
data_frame = data_frame.drop(['id', 'date', 'price', 'yr_renovated','zipcode','lat','long'], axis=1)
# Y = data_frame[['price']]
# x_train, x_test, y_train, y_test =  train_test_split(X,Y,train_size=0.7) # %70 ini test için ayır.
print(data_frame)
X = 0

def Hq(X , Q_all:list):
    hq = 0 
    # Önce satırı al dizi olarak 
    # Satırı enumerate ile (indis,veri)  al
    # Q[indis] * veri
    # ^y = q0 + q1x1 + q2x2 + . . . + qnxn
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
        
        total = total + ( Y[i][0]  - Hq(X=satir,Q_all = Q_all) )**2
         

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
            total = total + (Hq(X=satir,Q_all = Q_all) - Y[i][0] )
    else:
        for i, satir in enumerate(X): 
            total = total + ( Hq(X=satir,Q_all = Q_all) - Y[i][0] )*X[i][k-1]

    result = (1/satir_sayisi) * total 
    return result 



q_temp = []  # Q listesinin değişimini iterasyon sonuna kadar tutacağımız geçici liste
Q_all  = []
cost_all = []
# başlangıç değeri için kolon sayısı + 1 adet 1 ekleyeceğim
for i in range(len(X.columns)+1):
    Q_all.append(1)
    q_temp.append(1)


def Gd_with_all(X , Y , Q_all:list, alfa=0.1, epoch= 100 ):
    
    satir_sayisi = len(X)
    kolon_sayisi = X.shape[1]

    for i in range(epoch):

        temp_cost = J_2(X=x_test, Y=y_test, Q_all=Q_all)
        cost_all.append(temp_cost) 

        for j in range(kolon_sayisi+1):
            q_temp[j] = Q_all[j] - alfa * J_derivate_2(X=X, Y=Y, Q_all=Q_all, k=j)   
        

        Q_all = copy.deepcopy(q_temp)

        if i % 50 == 0:
            print(f"{i}. Maliyetim = ",temp_cost )
            # plt.scatter(x = range(len(cost_all)) ,y = cost_all , color="red")       
            # # tanım
            # plt.xlabel("epoch")
            # plt.ylabel("cost")
            # plt.title("Epoch - Cost")
            # plt.show()    
    

    print(Q_all)
    


hatalar = [] 
def percent_error():
    for i in range(len(x_test)):
        # print(f"Q[0]={Q[0]}  Q[1]={Q[1]}  Q[2]={Q[2]}")
        gercek = y_test.values[i]
        print(f"Gerçek {i}. veri = {gercek}")
        
        tahmin = Hq(X = x_test[i], Q_all = Q_all)
        
        # print("X verileri = ", x_test[i] ) 
        print(f"Tahmin {i}. veri = {tahmin}")

        aradaki_fark = (abs(gercek - tahmin) / gercek) * 100
        # print(f" %{aradaki_fark} hata")

        hatalar.append(aradaki_fark)
    print("Ortalama hatalar oranı %", (sum(hatalar)/len(hatalar)) )




#Gd_with_all(X=x_train , Y=y_train, Q_all=Q_all, alfa=0.5 , epoch=200)
#percent_error()
#print(Q_all)

    
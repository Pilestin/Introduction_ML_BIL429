import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split


data_frame = pd.read_csv("kc_house_data.csv")
print(data_frame.shape)
data_frame.head()
data_frame.info()

plt.scatter(x = data_frame[['sqft_lot']] , y=data_frame[["price"]] , color="red")  
 
# tanım
plt.xlabel("sqft_lot")
plt.ylabel("price")
plt.title("Arsa - Fiyat")
plt.show()

plt.scatter(x = data_frame['sqft_living'] , y=data_frame["price"] , color="green")       
# tanım
plt.xlabel("sqft_living")
plt.ylabel("price")
plt.title("Yaşam alanı - Fiyat")
plt.show()



X = data_frame[['sqft_living','sqft_lot']]
Y = data_frame[["price"]]
x_train, x_test, y_train, y_test =  train_test_split(X,Y,train_size=0.7) # %80 ini test için ayır.

print(type(x_train))


scaler = MinMaxScaler()
scaler.fit(x_train) 
# x train scale edilecek. Bu yüzden uygun hale getirildi.

x_train = scaler.transform(x_train)
x_test  = scaler.transform(x_test) 


#print(type(y_train)) # pandas tipi
#type(x_train)        # numpy tipi 

def J(X, Y, Q0, Q1, Q2 ):
    """ VERİLEN Q DEĞERLERİ İÇİN TÜM TABLOYU DENER VE GERÇEK Y İLE FARKIN KARELERİNİ TOPLAR. BÖYLECE MALİYETİ ÖLÇER """
    # X : x_test.values
        # X[i][0] : sqft_living  
        # X[i][1] : sqft_lot 

    # Y : y_test.values
    # hx = Q0 + Q1*x1 + Q2x2   
    
    total = 0
    m = len(X)
    for i in range(m):
        total = total + ( Q0 + Q1*X[i][0] + Q2*X[i][1] - Y[i][0])**2
    return 1/(2*m) * total 

    
    
def J_derivate(X, Y, Q0, Q1, Q2, k ):
    """GRADİENT D. ALGORİTMASI İÇİN MALİYETİN, TÜREVİNİN ALINDIKTAN SONRA, HESAPLANDIĞI KISIM"""
    # X : x_test.values
        # X[i][0] : sqft_living  
        # X[i][1] : sqft_lot 

    # Y : y_test.values
    # hx = Q0 + Q1*x1 + Q2x2  
    total = 0
    result = 0
    m = len(X)
    if(k==0):
        for i in range(m):
            
            total = total + (  Q0 + Q1*X[i][0] + Q2*X[i][1] - Y[i][0] )
    else:
        for i in range(m):
            
            total = total + ( Q0 + Q1*X[i][0] + Q2*X[i][1] - Y[i][0] )*X[i][k-1]

    result = (1/m) * total 
    return result 

Q    = [1,1,1]
# maliyetlerimizi bir listede tutalım
cost = []


def GradientDescent(Q, X , Y , alfa=0.1, epoch=10000):
    # epoch sayısı kadar tablomuzu tekrar tekrar gezip, modelimizi eğiteceğiz.
    for i in range(epoch+1):
        # Her döngüde geçici olarak maliyeti hesaplayalım ve bu değeri cost listemize atalım
        tempCost = J(X=x_test, Y=y_test.values, Q0=Q[0], Q1=Q[1], Q2=Q[2])
        cost.append(tempCost) 

        # Paramtere hesaplama kısmı 
        q0 = Q[0] - alfa * J_derivate(X=X, Y=Y, Q0=Q[0], Q1=Q[1], Q2=Q[2], k=0)
        q1 = Q[1] - alfa * J_derivate(X=X, Y=Y, Q0=Q[0], Q1=Q[1], Q2=Q[2], k=1)
        q2 = Q[2] - alfa * J_derivate(X=X, Y=Y, Q0=Q[0], Q1=Q[1], Q2=Q[2], k=2)
        
        # Paramterlerin eşzamanlı güncellenmesi
        Q[0] = q0
        Q[1] = q1
        Q[2] = q2

        # her 50 adımda maliyetin yazılması ve tablo çizilmesi 
        if i % 50 == 0:
            print(f"Maliyet[{i}] = ",cost[i] )
            print(f"Q : ",Q)
            plt.scatter(x = range(len(cost)) ,y = cost, color="green")       
            # tanım
            plt.xlabel("iterasyon")
            plt.ylabel("maliyet")
            plt.title("50 adımda bir Maliyet")
            plt.show()    


GradientDescent(Q=Q, X=x_train , Y=y_train.values, alfa=0.35 , epoch=8000)
# Proje-1: Regresyon

Bu proje iki aşamadan oluşmaktadır. İlk kısım lineer regresyon kullanarak, bir evin iki özelliği ile fiyatını tahmin etmeye yöneliktir. İkinci kısım ise evin bir çok özelliğini, çoklu lineer regresyon kullanarak fiyatını tahmin etmeye yöneliktir. 

Direk kodları incelemek için odev1.py ve odev2.py dosyalarını inceleyebilirsin. Eğer notlar ile beraber incelemek istersen proje_1_Regresyon.ipynb dosyasını inceleyebilirsin.


Verileri indirmek için : <a href="kc_house_data.csv">kc_house_data.csv</a>


Bu rapor Notion kullanarak yazılmıştır. Buradan Notion sayfasına ulaşabilir ve raporu daha düzenli okuyabilirsiniz.


<img src="https://upload.wikimedia.org/wikipedia/commons/4/45/Notion_app_logo.png" alt="https://upload.wikimedia.org/wikipedia/commons/4/45/Notion_app_logo.png" width="30px" /> Link : [Notion](https://www.notion.so/Proje-1-Regresyon-08c0dec28ff349439db3dfe26f1b0b5d)


Tüm işlemler google colab kullanarak yapıldı. Kodlara ve .ipynb dosyasına ulaşmak için : 


<img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" alt="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="30px" /> Link : [Proje 1 - Regresyon](https://github.com/Pilestin/My_ML_Adventure/tree/master/Makine%20%C3%96%C4%9Frenmesine%20Giri%C5%9F/Proje%201%20-%20Regresyon)


Veri Seti : **`https://www.kaggle.com/datasets/harlfoxem/housesalesprediction`**

## a) *Yaşam alanı (sqft_living) ve arsa (sqft_lot) özniteliklerini (feature, attribute) kullanarak evin fiyatını tahmin edecek basit bir regresyon modeli geliştiriniz.*

- İlk olarak yapmamız gerekenler gerekli kütüphaneleri import etmek olacak. Tabi ki modelimi eğiteceğimiz algoritmamızı kendimiz yazacağız. Fakat verileri uygun veri yapılarında tutmayı, görselleştirmeyi veya parçalamayı uygun kütüphaneleri kullanarak yapacağız.

Başlangıç olarak nump, pandas ve maplotlib kütüphanelerini import ettik. Yeri geldiğinde bir kaç import işlemi daha yapılacak 



![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled.png)

Verilerimizi .csv dosyasından almamız gerek. Bunun için pandas kütüphanesinden read_csv metodunu kullanıyoruz. Buradan bir pandas data frame tipinde nesne dönecektir ve bu nesne tüm tabloyu tutmakta. 

İlk olarak bu data frame nesnesinin shape( boyut ) bilgisine bakıyoruz. 

Ardından head ile ilk 5 satırı görmekteyiz.

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%201.png)

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%202.png)

- Verilerimizi inceledikten sonra artık train ve test olarak ayırma işlemine geçebiliriz.

Öncelikle ayırma işlemi için sklearn.model_selection içerisinden train_test_split metodunu import ettik. X ‘e attribute’larımızı aldık. Y ise çıkış değeri olan fiyat biligisini tutacak. 

Bu metodun içerisine X , Y ve train_size bilgisini verip verilerimizi ayırdık. 

x_train ve y_train ile verilerimizi eğğiteceğiz. x_test ve y_test ile ise doğrulama ve maliyet değerlerimizi kontrol edeceğiz.

Burada x ve y parçalarımızın her biri pandas Dataframe tipinde.  

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%203.png)

- Sıradaki işlemimize geçmeden önce verilerimize normalizasyon uygulamalıyız. Aksi halde verilerimizi eğitmemiz çok daha uzun sürecektir.

Bu işlemi kendimiz de yazabilirdik fakat burada hazır bulunan sklearn.preprocessing içerisindeki MinMaxScaler kullanımı tercih edildi.

Arkaplanda aşağıdaki formül her x değeri için uygulanmakta. 

<aside>
📏 $(X - min(x)) / ( max(x) - mix(x))$

</aside>

Normalize edilen veirler ise tekrar x_train ve x_test değişkenlerine verildi. 

Şu anda bu değişkenler **numpy.ndaray** tipinde. Bunların üzerinde işlem yapmamız artık daha kolay olacak. 

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%204.png)

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%205.png)

X verileri 

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%206.png)

Y verileri 

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%207.png)

- Şimdi elimizde hazır verilerimiz bulunduğuna göre kendi fonksiyonlarımızı yazabiliriz. İlk olarak maliyet fonksiyonumuzu görelim.

### Maliyet Fonksiyonu - 1

- Maliyet fonksiyonumuz x ve y aralarındaki hata oranını bize göstermekteydi. Ayrıca optimum’a ne kadar yaklaştığımızı bulabilmekteydik.
1. soru için J fonksiyonumuz sadece Q0 , Q1 , Q2 değerleri ile çalışacağı için direk bu parametreleri alabilir. Fakat 2. soruda bunları bir liste içerisinde tutacağız.

Bu fonksiyon ilk olarak bir toplam değişkeni belirlemekte. Ardından X ‘in boyutu kadar döngüde parantez içerisinde verilen işlemi uygulamakta.

Bu işlem aslında Hq(x) hipotez değerimizdir. Bunu 2. soruda ayrı bir fonksiyon olarak ayıracağız fakat şu an için bu şekilde devam edilecek.

Bu hipotez ile doğrusal bir eğri uydurmuş ve bu eğri ile gerçek Y değeri arasındaki farkı bulmuş oluruz. Yani hatamız. Ardından bu değerlerin her satır için bulunup karelerini alarak toplamaktayız.

Son olarak ise 1/2m yani veri sayısının 2 katına sonucumuzu bölmekteyiz. 

Maliyet fonksiyonumuz tamamlandı. 

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%208.png)

- Şimdi gradient descent içerisinde kullandığımız maliyet fonksiyonunun türevini yapan fonksiyonu yazabiliriz.

Burada gördüklerimiz de yukarıda yaptığımız işlemlerin neredeyse aynısı. Farkımız türev alındığı için kare bölümü çarpan olarak gelip 1/2m değerini 1/m ‘e çevirmekte. 

Ayrıca parantez içerisinin de türevi alınacağı için ( Dikkat Q’ya göre türev alınmakta. Bilinmeyen Q ) Q’ların önündeki katsayılar çarpan olarak gelmekte.

Son olarakta toplam değer 1/m ile çarpılmakta.

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%209.png)

- Şimdi asıl eğitme aşamasını yapacağımız GradientDescent algoritmamızı yazabiliriz. Ama öncelikle Q değerlerimizi oluşturalım.

Başlangıç için Q değerlerini 1,1,1 seçebiliriz. Fakat bu sayılar optimum değerlere doğru değişeceği için doğru seçim yapmak önemlidir. İlk uygulamada Q değerleri bazı sayılara yakınsamaktadır. Bu sayıların yakın olduğu değerleri vererek eğitme aşamasını kısaltmayı ve maliyet değerlerini düşürmeyi hedefleyebiliriz. 

Ayrıca cost listesi oluşturarak her iterasyonda maliyetin ne kadar olduğunu tutabiliriz.

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2010.png)

### GradientDescent Algoritması

Şimdi tüm parametrelerimizi değiştirdiğimiz ve maliyeti gördüğümüz aşamaya geldik.

Burada ilk olarak bir epoch değerimiz var. Bu değer kadar x’leri eğitmekteyiz. 

Ardından her adımda bir maliyet fonksiyonunu kullanarak hatamızı cost listesi içerisine atmaktayız. Burada hatalarımızı ezberleme olmaması için x_test ve y_test ile hesaplamaktayız.

Sonrasında her Q değerini q0,q1,q2 gibi değişkenlerde tutarak hesaplamaktayız. Bunu Q değerinden alfa (varsayılan 0.1 seçildi ) öğrenme katsayısı * J_derivate değerini çıkararak yapmaktayız. Matematiksel olarak 

$$
Qj := Qj - a * ((1/m)*∑(hq(x)-y)*x  
$$

işlemini yapmakta. Burada J_derivate ile gerekli veriler dışında k değerini de veriyoruz. Bu  k değeri türev ifadesinin ne olacağını belirlemektedir.

Bu işlem sonucunda Q[i] değeri “hatanın türevi * alfa” kadar miktar uzaklaşıp yakınlaşmakta. Q[i+1] değerinde bu değişim hesaplanırken yine Q[i] kullanıldığı için değişim tüm Q[] değerlerinin değişim hesaplandıktan sonra yapılmaktadır.

Yani eşzamanlı olarak değiştirilmekte ve böylece hesaplama hatası oluşmamaktadır. 

Son olarak ise her adımda maliyet bastırmak yerine her 50 adımda maliyet değeri bastırılmakta ve bu sonuçlar tablo olarak çizdirilmekte. 

Hemen alt satırda ise bu fonksiyon çalıştırılmakta. Farklı değerler için sonuçları çalıştırıp görelim.

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2011.png)

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2012.png)

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2013.png)

epoch 8000 , alfa =0.005 

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2014.png)

epoch 8000 , alfa =0.015 

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2015.png)

epoch 8000 , alfa =0.35, iterasyon 3400

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2016.png)

epoch 8000 , alfa =0.35 , iterasyon = 8000

- Epoch 8000 ve alfa 0.35 için Q değerlerimizi görelim.

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2017.png)

- Son olarak eğittiğimiz modelimizden Q değerlerini bulduk. Şimdi bu değerleri x_test ve y_test üzerinden Q değerlerimizi deneyelim ve yüzde kaç hata yaptığımızı her satır için bulalım

Öncelikle her x_test - y_test arasındaki yüzdelik hatayı bulacağımız için bu değerleri bir liste içerisinde tutmalıyız. Bu yüzden  hatalar listesi tanımladık 

Ardından degerlendirme fonksiyonunda her x_test satırını görmek için döngü yazdık. Bu döngüde gercek değişkenimiz y_test.values[i] ile o satırdaki, değerini bildiğimiz y bilgisini tutacak. 

tahmin değişkeni ise hipotezimizi uygulayacak. 

Şimdi aradaki farkın mutlak değerini gercek veriye bölüp 100 ile çarparak 

$$
(|gercek - tahmin| / gercek )* 100
$$

aradaki yüzdelik farkı bulmuş olduk. Bunu listemize ekleyebiliriz.

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2018.png)

Döngü bittiğinde elimizde hataların hepsi bulunacak. Bunların aritmetik ortalamasını alarak ortalamada ne kadar hatamız olduğunu bulabiliriz. 

Bu hücreyi de çalıştırırsak elde ettiğimiz sonuçlar şu şekilde ( Epoch 8000 ve alfa 0.35 için ) 

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2019.png)

<strong> Ortalama olarak %35 hata oranına sahibiz. </strong>

Şimdi 2. sorumuza geçelim 

## ***b) Veri setindeki bütün öznitelikleri kullanarak çok değişkenli bir regresyon modeli ile evin fiyatını tahmin ediniz.***

Bu bölümde önceki fonksiyon ve algoritmaları kullanacağımız için detaya girilmeden genelleştirilmiş halleri anlatılacaktır. 

İlk olarak verilerimizi X ve Y olarak tekrar parçalıyoruz. Burada evin fiyatını etkilemeyecek olan verileri ilik satırda atılmıştır.  Fakat öncekinin aksine parametre sayısından bağımsız genelleşmiş bir uygulama olacağı için eklenmesinde de tasarım açısından sorun yoktur. Sadece yavaşlamaya neden olacaktır. 

Bu yüzden bu değerler atılarak geri kalan 14 kolon attribute olarak seçildi. 

Hemen ardından veriler train ve test olarak %70 oranda parçalandı. 

Q değişkenlerimizi tutmak için Q_all[ ] listesi oluşturuldu.

Her iterasyonda bu Q_all[ *i* ] bilgisi değişeceği için bu değişimler tek tek değişkenlerde tutulması yanlış olacaktır. Bu yüzden q_temp listesi oluşturuldu.

 Bir döngü ile kolon sayısı + 1 kadar 1 değeri listelere eklendi. 

Bunun sebebi  x[*i*] kolon bilgilerine karşılık gelmekteydi. Örneğin sqft_living gibi. Fakat hipotezimizi $Q0 + Q1x1 + Q2x2 + . . . + Qnxn$  şeklinde düşündük. Burada fazladan Q0 fazlalığı bulunduğu için 15 adet 1 ekledik.

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2020.png)

- Şimdi verilerimizi Normalize edebiliriz.

Burada yaptığımız işlem ilk soruda yaptığımız ile aynı 

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2021.png)

- Önceki soruda hipotezimizi hep başka fonksiyonların içerisinde yazmıştık. Şimdi ise bunu ayıralım

Hq ile hipotezimizi bulacağız. Bunu 

$$
Q0 + Q1x1 + Q2x2 + . . . + Qnxn
$$

işlemi ile tanımlamıştık. Bu yüzden bu toplamı tutacak önce hq değişkenimizi oluşturuyoruz. Ardından her X satırı için bu formülü işletiyoruz. Burada for döngüsünde indisleri de kullanmak istediğimiz için enumerate ile X’leri key value haline getirdik.

Ayrıca Q[0] verimizin fazlalık olduğunu ve x çarpanı olmadığını belirtmiştik. Bu yüzden iterasyonu 1’den başlatıyoruz. 

Q[0] ‘ı ise en sonda sonucumuza ekleyip değeri döndürüyoruz. 

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2022.png)

### Maliyet Fonksiyonu - 2

- Artık daha genel maliyet ve gradiant fonksiyonları yazacağımız için bazı bölğmleri değiştireceğiz.

Burada satır sayımızı aldıktan sonra toplam ve result değişkenleri belirliyoruz. 

Burada tekrar enumerate kullanarak X içerisindeki satırlara ulaşıyoruz ve her satırı sayabilir oluyoruz. 

Artık ana işlemimiz olan farkın karelerini toplama işlemini yapabiliriz. 

Burada satır bilgisi elimizde olduğu için ( 14 attribute’a sahip 1 tane X bilgisi  ) bunu ve Q bilgilerini Hq fonksiyonumuza verip tahmini y çıktısını buluyoruz. 

Bunu Gerçek Y ile çıkartıp karesini alıyoruz ve sonuçları topluyoruz

En sonda bu toplamı 2*satır sayısına bölüyoruz. 

Burası gördüğümüz maliyet fonksiyonunun aynısı. 

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2023.png)

- Şimdi J_derivate ile türevli halini görelim

Burada ise yine aynı işlemleri yapıyor , Hq ile hipotezi hesaplayıp farkı buluyoruz. Öncesinde yaptığımız gibi yine X[i][k-1] katsayısı ile çarpıyoruz. (O Q[i] değerinin önündeki x attribute’u ) 

Ve en sonda toplam değeri satır sayısına bölüp döndürüyoruz. 

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2024.png)

### GradientDescent Algoritması

- Modelimizi eğitme aşamasını genel bir şekilde yazmaya geçebiliriz.

Burada ilk olarak copy kütüphanesini import ettik. Yeri geldiğinde bahsedilecek.

Ardından cost_all listesi oluşturuldu. Bu liste ile her iterasyonda maliyetimizi bu listeye ekleyeceğiz. Böylece listedeki verileri ve liste uzunluğunu kullanarak matplotlib’e tablo oluşturtacağız. 

Ardından döngü ile verilen epoch sayısı kadar modelimizi eğiticez. 

Her döngü içerisinde temp_cost ile maliyeti hesaplamaktayız ve bu veriyi cost_all  listemize eklemekteyiz. 

Ardından bir döngü ile modelin eğitildiği kısmı yapmaktayız. Burada kolon sayısının bir fazlası kadar döngü oluşturduk. Çünkü başta da belirttiğimiz gibi Q[0] yanında X çarpanı bulunmamakta.

Şimdi içerideki her iterasyonda Q[i] değerinin ne kadar değiştiği önceki soru ile benzer bir şekilde J_derivate_2 ile hesaplanacak. Ardından önceki sorudan farklı olarak bu geçici sonuç bir listede tutulacak. Bu işlem her Q değeri için yapıldıktan sonra içerideki döngümüz bitmiş olacak. 

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2025.png)

Döngü sonlandıktan sonra Q listemizi bulduğumuz geçici q_temp ile değiştirebiliriz. Fakat burada  Q_all = q_temp yaparsak python liste özelliğinden dolayı shallow copy yapmış olacağız ve referans kopyalanması sorunu oluşacak. Bu yüzden import ettiğimiz copy ile deepcopy metodunu kullanarak değerleri kopyalıyoruz. 

Bu iterasyonu her i değeri için görmek uzun bir süreç olacağı için her 50 adımda sonucu 

göreceğimiz bir ifade ekliyoruz. Burada ayrıca plt.scatter ile cost_all listemizi verip maliyetimizin nasıl bir grafik izlediğini de çizdiriyoruz.

Sonuçları ve eğitmeyi en sonda görelim 

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2026.png)

- Şimdi ilk soruda yaptığımız yüzdelik hata bulmayı revize edelim ve tekrar deneyelim.

Burada hataları tutacak bir listemiz var. Bu listeye her iterasyonda gerçek değer ile tahmin değer arasındaki yüzdelik farkı buluyoruz. Tahmini ise x_test ve Q değerlerini hipotez fonksiyonumuza vererek yaptık. 

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2027.png)

Şimdi sonuçlarımızı görelim.

### (alfa=0.20 , epoch=1000)

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2028.png)

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2029.png)

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2030.png)

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2031.png)

Q değerleri = [-154535.40088235194, -23601.05417902612, 382231.68690385355, 521983.1666968419, -8862.907305400215, 132577.3125777927, 471541.3542879963, 233011.74856392056, 69281.84846283196, 959837.6930524483, 592346.8325346287, 367966.8133860462, -404452.1103062461, 463869.4666701383, -40863.99019222211]

### %’lik hata

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2032.png)

### (alfa=0.005 , epoch=1000)

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2033.png)

### %lik hata

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2034.png)

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2035.png)

![Untitled](Proje-1%20Regresyon%2008c0dec28ff349439db3dfe26f1b0b5d/Untitled%2036.png)

# Sonuç :

<strong>
Veriler x_train ve y_train ile eğitilmiş ve x_test ve y_test ile maliyet değerleri kontrol edilip , hata oranları bulunmuştur.

İlk soruda 2 parametre için alfa ve epoch değerleri ile oynanarak %30 - % 50 arasında hata oranları alınmıştır

Fakat 2. soruda tüm parametrelerin verilmesinde hata oranları %99.99 oranında olmuştur.
</strong>

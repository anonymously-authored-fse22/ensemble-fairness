#!/usr/bin/env python
# coding: utf-8

# <b>Portekiz bankasından alınmış olan bu veri seti içeriğinde kurumun telefon görüşmeleri ile yapmış olduğu pazarlama kampanyaları detayları yer almaktadır. Bu telefon görüşmeleri sonucunda müşterinin "vadeli mevduat" ürününe sahip olup olmayacağı ("evet" ya da "hayır") tahminlenmeye çalışılmıştır. 
# Veri içeriğinde 45211 instances ve 17 attributes bulunmaktadır.<b>

# In[ ]:


import pandas as pd
import numpy as np


# ## Betimleyici Analitik Adımları

# In[ ]:


df = pd.read_csv("bank_marketing.csv")
df.head(10)


# <img src="11.png">

# <b>Veri tiplerini inceliyoruz<b>

# In[ ]:


df.info()


# In[ ]:


print("Bank marketing veri seti {rows} satır veri içermektedir.".format(rows = len(df)))


# <b>Verinin mod,medyan, ortalama gibi değerlerini inceliyoruz.<b>

# In[ ]:


df.describe()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,8))

sns.countplot(x = 'marital', hue = 'y', data=df, palette = 'inferno')

plt.show()


# <b> Üstteki görselde gördüğümüz üzere bankada vadeli para tutma oranları bekar insanlarda daha yüksek <b>

# In[ ]:


sns.distplot(df.age)


# <b>Bu görselde yaş değerlerinin dağılımını inceledik. İletişime geçilen müşterilerin yaşları 30 ile 40 arasında yoğunlaştığı gözleniyor. <b>

# In[ ]:


plt.figure(figsize=(12,8))

sns.countplot(x = 'job', hue = 'y', data=df,palette = 'inferno')

plt.xticks(rotation=45)
plt.show()


# <b>Yukarıdaki görselde meslek dağılımlarına göre kişilerin vadeli hesap adetlerini görüyoruz. Oran olarak baktığımızda öğrencilerin vadeli hesap adetlerinin yüksek olduğu gözleniyor, bunun yanı sıra mavi yakalı olarak ifade edilen çalışanların ise vadeli hesap oranının çok düşük olduğu görülmektedir. <b>

# In[ ]:


plt.figure(figsize=(8,8))

sns.countplot(x = 'contact', hue = 'y', data=df, palette = 'inferno')

plt.xticks(rotation=45)
plt.show()


# <b>Hücresel iletişim türü olan müşterilerin vadeli mevduat hesap adedi en yüksek olduğu görülüyor.<b>

# In[ ]:


numerik_degiskenler=df.describe().columns
df.hist(column=numerik_degiskenler,figsize=(20,20))
plt.show()


# <b>Bu görselde nümerik değişkenlerimizin tamamının histogramını inceledik.<b>

# In[ ]:


kategorik_degiskenler=df.describe(include=[object]).columns

fig, axes = plt.subplots(4, 3, figsize=(20, 20))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.7, hspace=0.3)

for i, ax in enumerate(axes.ravel()):
    if i > 7:
        ax.set_visible(False)
        continue
    sns.countplot(y = kategorik_degiskenler[i], data=df, ax=ax)
plt.show()


# <b>Bu görselde kategorik değişkenlerimizin tamamını çubuk grafikte inceledik.<b>

# In[ ]:


#age değerlerimizde outlier'lar var mı bakalım:

plt.figure(dpi=130)
sns.boxplot(df["age"])
plt.show()


# <b> Bu görselde yaş değişkenin outlier değerleri olduğunu tespit ettik.<b>

# ## Veri Temizleme ve İşleme Adımları

# <b>Her satır için eksik değerlerin yüzdesine baktığımızda null değer olmadığını görüyoruz. 
#    Eğer eksik değerler olsaydı bunları medyan, ortalama veya mod ile doldurmamız gerekecekti.<b>

# In[ ]:


missing_values = df.isnull().mean()*100

missing_values.sum()


# <b>interquartile range belirliyoruz <b>

# In[ ]:


q1 = df["age"].quantile(0.25)
q3 = df["age"].quantile(0.75)

iqr = q3 - q1

print("1st Quartile: " + str(q1))
print("3rd Quartile: " + str(q3))
print("Inter-quartile range: " + str(iqr))


# In[ ]:


#age kolonu için alt ve üst sınırlarımızı belirlelim : bu sınırlar dışındaki veriler OUTLIER'lardır. 
lower_age_bound = q1 - 1.5*iqr
upper_age_bound = q3 + 1.5*iqr

# 1.5 genel olarak kullanılan sayı, fakat siz değiştirebilirsiniz. 

print("Lower age bound: " + str(lower_age_bound))
print("Upper age bound: " + str(upper_age_bound))


# <b>IQR sonrası 10 yaş altı ve 70 yaş üstü değerlerin outlier olduğunu görüyoruz. Bu nedenle bu verileri kaldıracağız.<b>

# In[ ]:


df.age.isnull().sum()


# In[ ]:


# kabul edilebilir range'in dışındaki değerleri (outlier'Ları) NON yapacağız.
df.age = df.age.map(lambda x: x if lower_age_bound < x < upper_age_bound else np.nan)

df.age.isnull().sum()
#487 adet outlier verimiz olduğunu tespit ettik. Bu satırları veri setinden kaldıracağız.


# In[ ]:


df_new = df[~df['age'].isnull()]
df_new


# In[ ]:


print(len (df_new[df_new['pdays'] < 0] ) / len(df_new) * 100)
print(len (df_new[df_new['pdays'] > 400] ) / len(df_new) * 100)


# pdays kolonu müşteriyle bir önceki kampanya için iletişime geçildikten sonraki gün sayısını gösteriyor.İlgili kolonun detayına baktığımızda -1 içeren değerin yüzde 82lik büyük bir bölümü oluşturduğunu. 400 ve üzerindeki değerlerin ise %0.52 sini oluşturduğunu görüyoruz. 
# -1 muhtemelen müşteriyle daha önce iletişime geçilmediği veya eksik verileri temsil ettiği anlamına geliyor diye yorumladık. Bu kolonun modelimize bir faydası olmayacağını düşündüğümüz için kaldıracağız.

# <b> ML kısmına geçmeden önce veri setimizi işleme adımını yapmamız gerekmektedir. 
#     Veri setimizi incelediğimizde hedef değişken olan "y" içeriğindeki "yes", "no" değerlerine binary(0,1) hale getirmemiz gerekmektedir. 
#     Aynı şekilde cinsiyet, eğitim durumu vs içeren kategorik sütunları da dummy değişkenlere dönüştürmemiz gerekmektedir.<b>

# In[ ]:


def get_dummy_from_bool(row, column_name):
    
    return 1 if row[column_name] == 'yes' else 0
#sütun içeriği hayır ise 0, evet ise 1 döndür

def get_correct_values(row, column_name, threshold, df):
    
    if row[column_name] <= threshold:
        return row[column_name]
    else:
        mean = df[df[column_name] <= threshold][column_name].mean()
        return mean
#eğer değer threshold yani sınırın üzerindeyse ortalama değeri ver


# In[ ]:


#temizlenmiş datayı hazırlıyoruz
def clean_data(df):
    
    cleaned_df = df_new.copy() #ana veri setimizi kopyalıyoruz
    
    bool_columns = ['default','housing','loan','y']
    
    for bool_col in bool_columns:
        
        cleaned_df[bool_col + '_bool'] = df.apply(lambda row: get_dummy_from_bool(row, bool_col),axis=1)
    
    cleaned_df = cleaned_df.drop(columns = bool_columns)
    #evet ve hayır içeren değişkenlere 0-1 atıyoruz.
    
    cat_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    
    for col in  cat_columns:
        
        cleaned_df = pd.concat([cleaned_df.drop(col, axis=1),
                                pd.get_dummies(cleaned_df[col], prefix=col, prefix_sep='_',
                                               drop_first=True, dummy_na=False)], axis=1)
    #kategorik veri içeren değişkenler dummy değişkenlere dönüştürüldü
    
    cleaned_df = cleaned_df.drop(columns = ['pdays'])
    #pdays kolonunu kaldırmaya karar vermiştik
    

    return cleaned_df


# In[ ]:


cleaned_df = clean_data(df_new)
cleaned_df.head()


# In[ ]:


cleaned_df.info()


# <b>Vadeli yatırmak ile diğer değişkenler arasındaki pozitif korelasyona baktığımızda en yüksek öznitelikler şu şekilde;
# - "Duration" kolonu yani son telefon konuşmasının uzunluğu 0.39 ile en yüksek pozitif etki yaptığı gözlemleniyor. 
# - "poutcome success" yani bir önceki pazarlama kampanyasının başarılı sonuç vermesi 0.31 ile ikinci en yüksek pozitif etkiyi yaptığını görüyoruz.
# - Ev kredisi alınmaması da yüksek etki edenler arasında.
# - Ayrıca kampanya için aranan mart, eylül ve ekim ayları en çok pozitif etki yaratan aylar olarak göze çarpıyor.
#     <b>

# Not: Görsel biraz büyük olduğu için okunması zor ancak sadece bu veriyi analiz etmek bile kampanya hedefini belirleme aşamasında büyük rol oynayabilir.

# In[ ]:


plt.figure(figsize=(40,40))
sns.heatmap(cleaned_df.corr(),square=True,annot=True,cmap= 'Spectral')


# ## ML Bölümü - Tahminleyici Analitik Algoritmalarının Uygulanması

# ### <b> 1- Tahminleme algoritmalarında ilk olarak  <i>Random Forest Classifier</i> seçiyoruz.<b>

# Decision tree algoritmalarında en büyük problem aşırı öğrenme ve veriyi ezberlemektir. RF modeli ise bu problemi çözmek için hem veri setinden hem de öznitelik setinden rassal olarak farklı alt setler seçer ve bunları train eder. Farklı veri setleri üzerinde eğitim gerçekleştiği için overfitting problemi azalır. Sonunda ise problem regresyon ise decision tree tahminlerinin ortalaması, sınıflandırma ise tahminler arasında en çok oy alanı seçeriz.
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[ ]:


X = cleaned_df.drop('y_bool', axis = 1)
y = cleaned_df['y_bool']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3 , random_state = 101)


# Hedef değişkenimizi y ye atadık ve train_test_split ile veri setimizi train(%70) ve test(%30) veri setlerine ayırdık. Daha sonrasında RF modelimizi kuracağız.

# In[ ]:


rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                            random_state=0, max_features= 10, 
                            max_depth= 5)

rfc.fit(X_train, y_train)

rfc_predict_1 = rfc.predict(X_test)


# Modelimizi kurarken tree adedini 100, maksimum derinliğini 5 olarak set ettik. N_jobs -1 ise tüm CPU kullanılacak anlamına geliyor.
# Modelimize train olarak ayırdığımız veri setini fit ettikten sonra, test verileri tahminlemeye çalışıyoruz.

# In[ ]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import accuracy_score


print(classification_report(y_test, rfc_predict_1))

score = round(accuracy_score(y_test, rfc_predict_1),3)

cm1 = cm(y_test, rfc_predict_1) #confusion matrix (TP, TN , FP, FN)

sns.heatmap(cm1, annot=True, fmt=".1f", linewidths=.3, square = True, cmap = 'PuBu')

plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Accuracy Score: {0}'.format(score), size = 20)
plt.show()


# <b>Modelimizin başarı metriklerine bakmak için confusion matrix, precision, recall ve f-1 scorelarını hesapladık. 
#     <br>Accuracy score yani doğruluk oranına baktığımızda hiç görmediğimiz bir veri setinde %90 gibi bir oran yakalamışız. Burada overfitting var mı yok mu onu kontrol etmemiz gerekiyor, bunun için bir sonraki adımda cross-validation metodunu uygulamamız gerekmektedir. 
#     <br>Precision değerinin yüksek olması da model seçimlerinde önemli bir kriterdir. Burda yüzde 90 ve yüzde 77 ile aslında bu modelin veri setimize uygun olabileceği düşüncesi ortaya çıkıyor.
#     <br>
#       
#   0'ların 11.844'ünü doğru, 67 adedini ise hatalı tahminlemiş. 
#     
#   1'lerin ise; 221 adedini doğru, 1286 adedini hatalı tahminlemiş.
# 
# <br>
#    Veri setinde hedef değerlerimizde 1'ler 0'lara göre çok az olduğu için model 1'leri iyi tahminleyemiyor. 
# Bunu çözümlemek için veri setindeki bu dengesizliği aşağıdaki 2 yöntemden birini uygulayarak gidermemiz gerekiyor:
# 1.downsampling: y değeri 0 olan gözlemler içerisinden y değeri 1 olan gözlem sayısı kadar gözlem seçilerek yeni bir veriseti oluşturulur.
# 2.upsampling
# 

# ## DownSampling

# In[ ]:


y_is_equal_to_0_data = cleaned_df[cleaned_df.y_bool == 0]
y_is_equal_to_1_data = cleaned_df[cleaned_df.y_bool == 1]

print(len(y_is_equal_to_0_data), len(y_is_equal_to_1_data))


# In[ ]:


#Hedef değişkenimizin 0 olduğu gözlem sayısını azaltarak 5.000 seviyesine çekelim
from sklearn.utils import resample
downsampled0s = resample(y_is_equal_to_0_data, replace=False, n_samples=5000, random_state=101)

print(len(downsampled0s), len(y_is_equal_to_1_data))


# In[ ]:


data_downsampled = pd.concat([y_is_equal_to_1_data , downsampled0s])
data_downsampled.shape


# In[ ]:


X_down = data_downsampled.drop('y_bool', axis=1)
y_down = data_downsampled['y_bool']

X_train, X_test, y_train, y_test = train_test_split(X_down,y_down,test_size = 0.3 , random_state = 101)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                            random_state=0, max_features= 10, 
                            max_depth= 5)

rfc.fit(X_train, y_train)

rfc_predict_1 = rfc.predict(X_test)


# In[ ]:


print(classification_report(y_test, rfc_predict_1))

score5 = round(accuracy_score(y_test, rfc_predict_1),3)

cm5 = cm(y_test, rfc_predict_1) 

sns.heatmap(cm5, annot=True, fmt=".1f", linewidths=.3, square = True, cmap = 'PuBu')

plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Accuracy Score: {0}'.format(score5), size = 20)
plt.show()


# <b>Classification Report'ta görüldüğü üzere 0'ları tahminlemede biraz geriledik; başarı %84'e düşmüş. Ancak 1'leri tahminlemede oldukça ilerleme kaydetmiş olduk(%84),Downsampling öncesi %25ti. 
#   <br>Accuracy değerimiz ise %90 dan %84'e geriledi fakat kabul edilebilir bir oran bu da.
# <br>Confusion Matrix'e göre modelimiz; 
#   
#   0'ların 1.247'sini doğru, 256 adedini ise hatalı tahminlemiş. 
#     
#   1'lerin ise; 1.284 adedini doğru, 235 adedini hatalı tahminlemiş.
# 
#   Görüldüğü üzere modelimiz artık 1'leri de daha doğru tahminler durumu geldi.

# ## Cross-Validation

# Modelimizin yüksek performansının rastgele olup olmadığınız görmemiz için cv uygulayacağız.

# In[ ]:


from sklearn.model_selection import cross_val_score

print(cross_val_score(RandomForestClassifier(n_estimators = 100, 
                                             n_jobs=-1, 
                                             random_state=0, 
                                             max_features= 10, 
                                             max_depth= 5), 
                       X_train, y_train, cv=5
                      )
     )


# In[ ]:


print('Mean of cv-scores: {0}'.format(round(np.mean(cross_val_score(RandomForestClassifier(n_estimators=100, 
                                                                                           n_jobs=-1, 
                                                                                           random_state=101, 
                                                                                           max_features= 10, 
                                                                                            max_depth= 5), 
                                                                     X_train, y_train, cv=5)
                                                    ),3)
                                        )
     )


# <b>Modelimizi 5 farklı eğitim ve test veri setine böldüğümüzde ortalaması %83.2 çıktı. 
#     <br>Yani RFC'da bulmuş olduğumuz %84 doğruluk oranı overfitting'den kaynaklı olmadığını görmüş olduk. Modelimiz gayet iyi çalışmış.
# <b>

# ### <b>2- Tahminleme algoritmalarında ikinci olarak  <i>XGBoost'u </i> seçiyoruz.<b>

# In[ ]:


from xgboost import XGBClassifier


# XGboost algoritması gradient boosting algoritmasının çeşitli düzenler sonrası yüksek performans gösteren şeklidir. Bir decision tree algoritmasıdır.

# In[ ]:


X = cleaned_df.drop('y_bool', axis = 1)
y = cleaned_df['y_bool']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 101)


# XGboost modeli için train ve test veri setlerini oluşturuyoruz ve parametrelerini belirliyoruz.
# <br> 
# XGB model ayarları için bir çok parametre mevcut. Bizim deneme yanılmalar sonrası en uygun bulup kullanmış olduklarımız şu şekilde <br> 
# -- n_estimators -> modelde kurulacak ağaç sayısı<br> 
# -- max_depth -> ağacın derinliğini ifade eder<br> 
# -- colsample_bytree -> her bir ağacı oluştururken sütunların oluşturduğu alt veri setleridir,default u 1 dir<br> 
# -- subsample -> eğitim örneklerinin alt örneklere oranı. Bu değeri 0.5 olarak ayarlamak XGBoost ağaçlarını büyütmeden önce eğitim verilerinin yarısını rastgele train edeceği anlamına gelir. Bu da overfitting i önler.

# In[ ]:


xgb = XGBClassifier(n_estimators=100, subsample=0.5,colsample_bytree=1, max_depth= 5)

xgb.fit(X_train,y_train.squeeze().values)

xgb_predict_1 = xgb.predict(X_test)


# In[ ]:


print('XGB accuracy score: %.3f' % (accuracy_score(y_test, xgb_predict_1)))


# In[ ]:


print(classification_report(y_test, xgb_predict_1))

score2 = round(accuracy_score(y_test, xgb_predict_1),3)

cm2 = cm(y_test, xgb_predict_1) 

sns.heatmap(cm2, annot=True, fmt=".1f", linewidths=.3, square = True, cmap = 'Greens')

plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Accuracy Score: {0}'.format(score2), size = 20)
plt.show()


# <b>Modelimizin başarı metriklerine bakmak için confusion matrix, precision, recall ve f-1 scorelarını hesapladık. 
#     <br>Accuracy score yani doğruluk oranına baktığımızda %91 gibi bir oran yakalamışız. 
#     <br>Precision değerinin yüksek olması da model seçimlerinde önemli bir kriterdir. Burda vadeli mevduat almayacakları tahminlemede yüzde 94 ile çok iyi bir oran yakalamışız ancak vadeli mevduat alacakları tahminlemede yüzde 61 ile düşük bir oran gelmiş.
#     <br>  
#   0'ların 11.458'ini doğru, 453 adedini ise hatalı tahminlemiş. 
#     
#   1'lerin ise; 715 adedini doğru, 792 adedini hatalı tahminlemiş.
#      <br> Bu modelimizde de 1 lerin oranını arttırmak için downsampling deneyebiliriz.
#         

# In[ ]:


y_is_equal_to_0_data = cleaned_df[cleaned_df.y_bool == 0]
y_is_equal_to_1_data = cleaned_df[cleaned_df.y_bool == 1]

print(len(y_is_equal_to_0_data), len(y_is_equal_to_1_data))

from sklearn.utils import resample
downsampled0s = resample(y_is_equal_to_0_data, replace=False, n_samples=5000, random_state=101)

print(len(downsampled0s), len(y_is_equal_to_1_data))


# In[ ]:


data_downsampled = pd.concat([y_is_equal_to_1_data , downsampled0s])
data_downsampled.shape


# In[ ]:


X_down = data_downsampled.drop('y_bool', axis=1)
y_down = data_downsampled['y_bool']

X_train, X_test, y_train, y_test = train_test_split(X_down,y_down,test_size = 0.3 , random_state = 101)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

xgb = XGBClassifier(n_estimators=100, subsample=0.5,colsample_bytree=1, max_depth= 5)

xgb.fit(X_train,y_train.squeeze().values)

xgb_predict_1 = xgb.predict(X_test)

print('XGB accuracy score: %.3f' % (accuracy_score(y_test, xgb_predict_1)))


# In[ ]:


print(classification_report(y_test, xgb_predict_1))

score6 = round(accuracy_score(y_test, xgb_predict_1),3)

cm6 = cm(y_test, xgb_predict_1) 

sns.heatmap(cm6, annot=True, fmt=".1f", linewidths=.3, square = True, cmap = 'Greens')

plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Accuracy Score: {0}'.format(score6), size = 20)
plt.show()


# <b>Classification Report'ta görüldüğü üzere 0'ları tahminlemede biraz geriledik; başarı %85'e düşmüş. Ancak 1'leri tahminlemede oldukça ilerleme kaydetmiş olduk(%85),Downsampling öncesi %53 tü. 
#   <br>Accuracy değerimiz ise %91 den %85'e geriledi fakat kabul edilebilir bir oran bu da.
# <br>Confusion Matrix'e göre modelimiz; 
#   
#   0'ların 1.259'unu doğru, 244 adedini ise hatalı tahminlemiş. 
#     
#   1'lerin ise; 1.314 adedini doğru, 205 adedini hatalı tahminlemiş.
# 
#   Görüldüğü üzere modelimiz artık 1'leri de daha doğru tahminler durumu geldi.

# ### <b>3- Tahminleme algoritmalarında son olarak  <i>Linear Support Vector Classifier</i> seçiyoruz.<b>

# In[ ]:


from sklearn.model_selection import train_test_split

X = cleaned_df.drop('y_bool', axis = 1)
y = cleaned_df['y_bool']

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.3, shuffle=True, stratify = y, random_state = 0)


# Veri setimizin %30'unu test için ayırdık. 
# Stratify: Bu parametreye hedef değişkenimi (y) değerini giriyorum. Bu şekilde böldüğüm tüm veri gruplarında y'nin oranının veri setim ile aynı olmasını sağlıyorum. 
# Bu stratify'ı train_test_split'i kullanırken belirlemeliyim. 
# Her çalıştırdığımızda train ve test verileri değişmesin diye random_state'e bir değer girdik.

# In[ ]:


#Normalizasyon – Feature Scaling yapıyoruz (Fitting)
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# <b>Scikit-Learn kütüphanesinden svm modülünü import ederek classifier nesnemizi tanımlıyoruz

# In[ ]:


from sklearn.svm import SVC

classifier = SVC(kernel='linear', random_state = 0)
classifier.fit(X_train, y_train)


# <b>Ayırdığımız test setimizi (X_test) kullanarak oluşturduğumuz model ile tahmin yapalım  ve elde ettiğimiz değerler (y_pred) ile hedef değişken (y_test) test setimizi karşılaştıralım

# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


print(classification_report(y_test, y_pred))

score3 = round(accuracy_score(y_test, y_pred),3)

cm3 = cm(y_test, y_pred) #confuison matrix

sns.heatmap(cm3, annot=True, fmt=".1f", linewidths=.3, square = True, cmap = 'coolwarm_r')

plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Accuracy Score: {0}'.format(score3), size = 20)
plt.show()


# Classification Report'a göre; Accuracy değerimiz %89.6. F1 scorelara baktığımızda ise 0'ları %94 başarıyla tahmin ediyor. Fakat 1'leri tahmin etmekte çok başarılı değil (%29).
# Veri setinde hedef değerlerimizde 1'ler 0'lara göre çok az olduğu için model 1'leri iyi tahminleyemiyor. 
# Bunu çözümlemek için veri setindeki bu dengesizliği aşağıdaki 2 yöntemden birini uygulayarak gidermemiz gerekiyor:
# 1.downsampling: y değeri 0 olan gözlemler içerisinden y değeri 1 olan gözlem sayısı kadar gözlem seçilerek yeni bir veriseti oluşturulur.
# 2.upsampling

# ## <b>DOWNSAMPLING  
#     Veri setimizi dengelemek için burada downsapling yöntemini uygulamayı uygun gördük

# In[ ]:


y_is_equal_to_0_data = cleaned_df[cleaned_df.y_bool == 0]
y_is_equal_to_1_data = cleaned_df[cleaned_df.y_bool == 1]

print(len(y_is_equal_to_0_data), len(y_is_equal_to_1_data))


# <b> Hedef değişkenimizin 0 olduğu gözlem sayısını azaltarak 5.000 seviyesine çekelim

# In[ ]:


from sklearn.utils import resample
downsampled0s = resample(y_is_equal_to_0_data, replace=False, n_samples=5000, random_state=101)

print(len(downsampled0s), len(y_is_equal_to_1_data))


# In[ ]:


data_downsampled = pd.concat([y_is_equal_to_1_data , downsampled0s])
data_downsampled.shape


# <b>Şuan hedef değişkenimize göre daha dengeli dağılım gösteren bir veriseti elde etmiş olduk. Şimdi yukarıda uyguladığımız adımları yeniden uygulayarak tekrar Linear SVC modelini kuracağız

# In[ ]:


X_down = data_downsampled.drop('y_bool', axis=1)
y_down = data_downsampled['y_bool']

X_train, X_test, y_train, y_test = train_test_split(X_down, y_down , test_size = 0.3, shuffle=True, stratify = y_down, random_state = 101)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

classifier = SVC(kernel='linear', random_state = 0)
classifier.fit(X_train, y_train)
y_pred_2 = classifier.predict(X_test)


# <b> Classification Report ve Confusion Matrix e tekrar bakalım 

# In[ ]:


print(classification_report(y_test, y_pred_2))

score4 = round(accuracy_score(y_test, y_pred_2),3)

cm4 = cm(y_test, y_pred_2) 

sns.heatmap(cm4, annot=True, fmt=".1f", linewidths=.3, square = True, cmap = 'coolwarm_r')

plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Accuracy Score: {0}'.format(score4), size = 20)
plt.show()


# <b>Classification Report'ta görüldüğü üzere 0'ları tahminlemede biraz geriledik; başarı %84'e düşmüş. Fakat 1'leri tahminlemede oldukça ilerleme kaydetmiş olduk(%84). Accuracy değerimiz ise %90 dan %84'e geriledi.
# <br>Confusion Matrix'e göre modelimiz; 
#   
#   0'ların 1.258'ini doğru, 242 adedini ise hatalı tahminlemiş. 
#     
#   1'lerin ise; 1.283 adedini doğru, 239 adedini hatalı tahminlemiş.
# 
#   Görüldüğü üzere modelimiz artık 1'leri de daha doğru tahminler durumu geldi.

# # Comparison ML Models

# <b> Problemimizin içeriğini tekrar hatırlatmak gerekirse; 
#     <br>Portekiz bankasından alınmış olan veri setinde kurumun telefon görüşmeleri ile yapmış olduğu pazarlama kampanyaları detayları yer almaktadır. Bu telefon görüşmeleri sonucunda müşterinin "vadeli mevduat" ürününe sahip olup olmayacağı ("evet" ya da "hayır") tahminlenmeye çalışılmıştır. Problemimizin girdi ve çıktıları betimleyici analitik kısmında detaylı anlatılmıştır.
#     <br> Betimleyici analitik, veri temizleme ve işleme adımlarını tamamladıktan sonra müşterinin "vadeli mevduat" ürününe sahip olup olmayacağını üç farklı tahminleme modeli ile çalıştık. 
#     <br> Bu modeller; 
#     <i><br> 1) Random Forest Classifier
#     <br> 2) Gradient Boosting (XGBoost Classifier)
#     <br> 3) Linear Support Vector Classifier</i>

# <b>1)
# <img src="RFC.png">

# <b>2)
# <img src="XGBoost.png">

# <b>3)
# <img src="LS.png">

# Kullanmış olduğumuz modelleri karşılaştırırken accuracy, precision ve F1 değerlerini inceliyoruz.
# <br>Accuracy en yüksek <b>%85.1</b> ile <i>XGBoost</i> algoritması oluyor, daha sonra <b>84.1</b> <i>Linear SVC</i> ve <b>83.8</b> <i>Random Forest</i> geliyor.
# 
# <br> F1 değerlerine baktığımızda ise <b>%85</b> ile <i>XGBoost</i> yine en iyi tahminde bulunan model oluyor.

# <b>Sonuç olarak Gradient Boosting XGBoost modeli, potansiyel bir müşterinin vadeli mevduat açtırıp açtırmayacağını tahmin etmek için en iyi model olduğuna karar veriyoruz.

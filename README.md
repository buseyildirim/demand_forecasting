# Demand Forecasting
### İş Problemi :
İş Problemi
Bir mağaza zinciri, 10 farklı mağazası ve 50 farklı ürünü için 3 aylık bir talep tahmini istemektedir.

### Veri Seti
Bir mağaza zincirinin 5 yıllık verilerinde 10 farklı mağazası ve 50 farklı ürünün bilgileri yer almaktadır.

### Değişkenler

<table>
  <tr >
    <th>Değişken</th>
    <th>Açıklama</th> 
  </tr>
    <tr>
    <td>date</td>
    <td>Satış verilerinin tarihi</td> 
  </tr>
  
  <tr>
    <td>Store</td>
    <td>Mağaza ID'si</td> 
  </tr>
  <tr>
    <td>Item</td>
    <td>Ürün ID'si</td> 
  </tr>
    <td>Sales</td>
    <td>Satılan ürün sayıları/td> 
  </tr>
</table>

### Projede yapılanlar :

▪ Random Noise <br>
▪ Lag/Shifted Features<br>
▪ Rolling Mean Features<br>
▪ Exponentially Weighted Mean Features <br>
kullanarak zaman serileri için trend, seasonality gibi değişkenler eklenmiştir.<br>
▪ Custom Cost Function (SMAPE)<br>
Lgbm için custom cost fonksiyon tanımlanmıştır.<br>
▪ LightGBM ile Model Validasyonu<br>

### Kaynak :

 Store Item Demand Forecasting Challenge
 https://www.kaggle.com/c/demand-forecasting-kernels-only


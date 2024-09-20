import pandas as pd
import yfinance as yf
from nixtla import NixtlaClient
import os

timegpt_api_key = "nixtla-tok-G4T0OjKtqp9nuPDajgRnmHqEcNUCWhgG409f6CMCV0sssGixLQLgtdJD7P03AzLEUhS73TUeVPqsWWCD"

nixtla_client = NixtlaClient(api_key=timegpt_api_key)

# Amazon hisse senedi fiyatları
ticker = 'AMZN'
amazon_stock_data = yf.download(ticker)
amazon_stock_data = amazon_stock_data.reset_index()

print(amazon_stock_data.head())

# Zaman serisi verisini görselleştirme
nixtla_client.plot(amazon_stock_data, time_col='Date', target_col='Close')

model = nixtla_client.forecast(
    df=amazon_stock_data,
    model="timegpt-1",
    h=24,  # (24 gün)
    freq="B",  # iş günü
    time_col="Date",
    target_col="Close",
)

print(model.tail())

# Gerçek ve tahmin edilen veriyi görselleştirme
nixtla_client.plot(
    amazon_stock_data,
    model,
    time_col="Date",
    target_col="Close",
    max_insample_length=60, 
)


import matplotlib.pyplot as plt
import pandas as pd

# Tahmin sonuçlarını veri çerçevesine ekleme
forecast_data = pd.DataFrame({
    'Date': model['Date'],
    'TimeGPT': model['TimeGPT']
})
amazon_stock_data = pd.concat([amazon_stock_data, forecast_data], ignore_index=True)

# Gerçek ve tahmin edilen veriyi görselleştirme
plt.figure(figsize=(14, 7))
plt.plot(amazon_stock_data['Date'], amazon_stock_data['Close'], label='Gerçek Kapanış Fiyatı', marker='o')
plt.plot(amazon_stock_data['Date'], amazon_stock_data['TimeGPT'], label='Tahmin Kapanış Fiyatı', marker='x')

plt.title('Amazon Hisse Senedi Fiyatları - Gerçek ve Tahmin')
plt.xlabel('Tarih')
plt.ylabel('Fiyat')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()


import numpy as np

# Tahmin hatasını hesaplama
actual_data = amazon_stock_data.set_index('Date')
forecast_data = forecast_data.set_index('Date')

# Gerçek kapanış fiyatlarını tahmin sonuçlarına ekleme
forecast_data['Gerçek'] = actual_data['Close']

# Hata
forecast_data['Hata'] = forecast_data['Gerçek'] - forecast_data['TimeGPT']

# Anormallik tespiti için eşik değeri belirleme
hata_std = np.std(forecast_data['Hata'])
anormallik_eşik = 2 * hata_std

# Anormallikleri belirleme
forecast_data['Anormal'] = np.abs(forecast_data['Hata']) > anormallik_eşik

anormallik_sayısı = forecast_data['Anormal'].sum()
anormallik_tarihleri = forecast_data[forecast_data['Anormal']].index

print(f"Toplam Anormallik Sayısı: {anormallik_sayısı}")
print("Anormal Olarak Tespit Edilen Tarihler:")
print(anormallik_tarihleri)

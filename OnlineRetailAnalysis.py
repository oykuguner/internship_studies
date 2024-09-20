import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

class OnlineRetailAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
    
    def load_data(self):
        try:
            self.df = pd.read_excel(self.file_path)
            print("Veri yüklendi.")
            print(self.df.head())
        except Exception as e:
            print(f"Veri yüklenirken hata oluştu: {e}")
    
    def check_missing_values(self):
        #Eksik veriler kontrol
        if self.df is not None:
            missing_values = self.df.isnull().sum()
            print("Eksik veri sayısı:\n", missing_values)
        else:
            print("Veri yüklenmedi.")
    
    def clean_data(self):
        #Eksik müşteri kimliklerini temizleme
        if self.df is not None:
            self.df = self.df.dropna(subset=['CustomerID'])
            print("Eksik müşteri kimlikleri temizlendi.")
            print(f"Temizlenmiş veri boyutu: {self.df.shape}")
        else:
            print("Veri yüklenmedi.")
    
    def data_summary(self):
        #Veri türlerini ve özet istatistikleri gösterme
        if self.df is not None:
            print("Veri Türleri:\n", self.df.dtypes)
            print("\nÖzet İstatistikler:\n", self.df.describe())
        else:
            print("Veri yüklenmedi.")
    
    def visualize_data(self):
        #görselleştirme
        if self.df is not None:
            plt.figure(figsize=(10, 6))
            sns.countplot(x='Country', data=self.df, order=self.df['Country'].value_counts().index)
            plt.xticks(rotation=90)
            plt.title('Ülkelere Göre Satış Dağılımı')
            plt.show()
        else:
            print("Veri yüklenmedi.")

file_path = r'C:\Users\Öykü Güner\OneDrive\Masaüstü\Online Retail.xlsx'

analysis = OnlineRetailAnalysis(file_path)  # Sınıfı başlat
analysis.load_data()  # Veriyi yükle
analysis.check_missing_values()  # Eksik verileri kontrol et
analysis.clean_data()  # Eksik verileri temizle
analysis.data_summary()  # Veri özetini göster
analysis.visualize_data()  # Veriyi görselleştir


reference_data, target_data = train_test_split(analysis.df, test_size=0.5, random_state=42)  #veriyi ikiye bölme

print("Referans Veri Boyutu:", reference_data.shape)
print("Hedef Veri Boyutu:", target_data.shape)

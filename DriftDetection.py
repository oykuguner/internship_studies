import numpy as np
from alibi_detect.cd import ChiSquareDrift
from OnlineRetailAnalysis import OnlineRetailAnalysis

file_path = r'C:\Users\Öykü Güner\OneDrive\Masaüstü\Online Retail.xlsx'

analysis = OnlineRetailAnalysis(file_path)
analysis.load_data()  #veriyi yükle
analysis.check_missing_values()  # Eksik verileri kontrol et
analysis.clean_data()  # Eksik verileri temizle

class DriftDetection:
    def __init__(self, reference_data, target_data):
        # Drift detection için veri setlerini ayarlama
        self.reference_data = reference_data
        self.target_data = target_data
        self.cd = None  # Drift detection modeli

    def prepare_data(self, column_name):
        #Veri setlerini kategorik bir sütun için numpy dizisine dönüştürme
        reference = self.reference_data[column_name].values
        target = self.target_data[column_name].values
        return reference, target
    
    def detect_drift(self, column_name):
        #Belirli bir sütun üzerinde drift tespit etme
        reference, target = self.prepare_data(column_name)
        self.cd = ChiSquareDrift(x_ref=reference, p_val=0.05)  # p değeri eşik değeri
        preds = self.cd.predict(target)  # Hedef veri üzerinde drift kontrolü
        print(f"{column_name} Sütununda Sapma Tespiti:")
        print(f"Drift Tespit Edildi: {preds['data']['is_drift']}, p-değeri: {preds['data']['p_val']}")

reference_data = analysis.df.sample(frac=0.5, random_state=42)  #veri setinin yarısını referans olarak al
target_data = analysis.df.drop(reference_data.index)  #kalan yarısını hedef veri olarak al

drift_detector = DriftDetection(reference_data, target_data)
drift_detector.detect_drift('UnitPrice')

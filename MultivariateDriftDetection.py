import pandas as pd
import numpy as np
from alibi_detect.cd import KSDrift
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

file_path = r'C:\Users\Öykü Güner\OneDrive\Masaüstü\Online Retail.xlsx'
df = pd.read_excel(file_path)

# Eksik verileri temizleme
df.dropna(subset=['CustomerID'], inplace=True)

# Özellikler seçimi ve ön işleme
features = ['Quantity', 'UnitPrice', 'CustomerID']
df_features = df[features]

# Kategorik verileri sayısal verilere dönüştürme
le = LabelEncoder()
df_features['CustomerID'] = le.fit_transform(df_features['CustomerID'].astype(str))

# Veriyi referans ve hedef olarak ayırma
reference_data = df_features.sample(frac=0.5, random_state=42)
target_data = df_features.drop(reference_data.index)

# Pandas veri çerçevelerini numpy dizilerine dönüştürme
reference_data_np = reference_data.to_numpy()
target_data_np = target_data.to_numpy()

print("Veri başarıyla yüklendi.")



class MultivariateDriftDetection:
    def __init__(self, reference_data, target_data):
        self.reference_data = reference_data
        self.target_data = target_data
        self.cd = None

    def detect_drift(self):
        # Drift testi için KSDrift modelini oluşturma
        self.cd = KSDrift(x_ref=self.reference_data)
        # Testi hedef veri üzerinde uygulama
        preds = self.cd.predict(self.target_data)
        return preds

# Drift tespiti
drift_detector = MultivariateDriftDetection(reference_data_np, target_data_np)
results = drift_detector.detect_drift()

print("Çok değişkenli sapma tespiti:")
print(f"Drift Tespit Edildi: {results['data']['is_drift']}, p-değeri: {results['data']['p_val']}")

# Görselleştirme
def plot_distributions(ref_data, target_data, feature_names, sample_size=50000):
    # Küçük örneklem
    ref_sample = ref_data[np.random.choice(ref_data.shape[0], sample_size, replace=False)]
    target_sample = target_data[np.random.choice(target_data.shape[0], sample_size, replace=False)]

    plt.figure(figsize=(15, 5))
    for i, feature in enumerate(feature_names):
        plt.subplot(1, len(feature_names), i + 1)
        plt.hist(ref_sample[:, i], bins=30, alpha=0.5, label='Reference Data')
        plt.hist(target_sample[:, i], bins=30, alpha=0.5, label='Target Data')
        plt.title(f'Distribution of {feature}')
        plt.legend()
    plt.tight_layout()
    plt.show()
# Özellik adları
feature_names = features

# Dağılımları çizme
plot_distributions(reference_data_np, target_data_np, feature_names)



import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

excel_dosyasi = r'C:\Users\Öykü Güner\OneDrive\Masaüstü\Online Retail.xlsx' 

data = pd.read_excel(excel_dosyasi)
numeric_data = data.select_dtypes(include=[np.number])

numeric_data_cleaned = numeric_data.dropna()

# Veri setini referans ve hedef veri setlerine ayırın
n = len(numeric_data_cleaned) // 2
reference_data = numeric_data_cleaned.iloc[:n]
target_data = numeric_data_cleaned.iloc[n:]

# PCA modelini referans verileri ile eğit
pca = PCA(n_components=2)
pca.fit(reference_data)

# Referans ve hedef verileri PCA uzayına projekte etme
reference_proj = pca.transform(reference_data)
target_proj = pca.transform(target_data)

# Orijinal verinin yeniden oluşturulmasındaki hata
def reconstruction_error(original_data, projection, pca_model):
    reconstructed = pca_model.inverse_transform(projection)
    error = np.mean((original_data - reconstructed) ** 2, axis=1)
    return error

# Referans ve hedef verileri için yeniden oluşturma hatası
reference_error = reconstruction_error(reference_data, reference_proj, pca)
target_error = reconstruction_error(target_data, target_proj, pca)

# Drift tespitini görselleştirme
plt.figure(figsize=(10, 5))
plt.plot(reference_error, label='Reference Error', color='blue')
plt.plot(np.arange(len(reference_error), len(reference_error) + len(target_error)), target_error, label='Target Error', color='orange')
plt.axvline(x=len(reference_error), color='red', linestyle='--', label='Drift Start')
plt.xlabel('Samples')
plt.ylabel('Reconstruction Error')
plt.title('PCA Reconstruction Error for Multivariate Drift Detection')
plt.legend()
plt.show()

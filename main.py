import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import random

# 1. VERİ YÜKLEME
try:
    # Dosya adını tam olarak belirttiğin gibi kullanıyorum
    df = pd.read_excel('train_v7.xlsx')
    print("Veri başarıyla yüklendi. Satır sayısı:", len(df))
    print(df.head())
except FileNotFoundError:
    print("HATA: 'train_v7.xlsx' dosyası bulunamadı. Lütfen dosyanın aynı klasörde olduğundan emin olun.")
    # Hata durumunda devam etmemek için burada duruyoruz ama sen kendi ortamında çalıştırabilirsin.
    raise SystemExit

# 2. VERİ ÖN İŞLEME
# Tur Zamanı formatını kontrol edelim. Eğer string ise (örn: "6:30.70") saniyeye çevirelim.
def time_to_seconds(time_str):
    if isinstance(time_str, str):
        parts = time_str.split(':')
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    else:
        return time_str # Zaten sayıysa dokunma

df['Tur_Zaman_Saniye'] = df['Tur Zamanı'].apply(time_to_seconds)

# Özellikler (Features) ve Hedef (Target) belirleme
# Metin sütunlarını (Araç Modeli) çıkarıyoruz, sadece sayısal/kategorik kodlanmış sütunları alıyoruz.
features_cols = ['Yıl', 'Motor', 'Güç (HP)', 'Tork (Nm)', 'Şanz.', 'Vites', 'Karoseri', 'Ağırlık', 'Çekiş', 'Aks Mesafesi (mm)', 'Ön Lastik (mm)', 'Arka Lastik (mm)', 'Yarış Arabası']
target_col = 'Tur_Zaman_Saniye'

X = df[features_cols].values
y = df[target_col].values.reshape(-1, 1)

# 3. ÖLÇEKLENDİRME (Normalization)
# Derin öğrenmede verilerin ölçeği çok önemlidir.
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 4. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# PyTorch Tensorlarına çevirme
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# DataLoader oluşturma
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 5. MODEL MİMARİSİ (Deep Learning)
class NurburgringNet(nn.Module):
    def __init__(self, input_dim):
        super(NurburgringNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        # self.network = nn.Sequential(
        #     nn.Linear(input_dim, 64),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(64),
        #     nn.Dropout(0.2),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(32),
        #     nn.Dropout(0.2),
        #     nn.Linear(32, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 1)
        # )
    
    def forward(self, x):
        return self.network(x)

input_dim = X_train.shape[1]
model = NurburgringNet(input_dim)

# 6. EĞİTİM AYARLARI
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 100

print("\n--- MODEL EĞİTİLİYOR ---")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')

# 7. TEST VE TAHMİN (5 Rastgele Araç)
print("\n--- 15 RASTGELE ARAÇ İLE TEST ---")
model.eval()
with torch.no_grad():
    # Tüm veri setinden rastgele 5 indeks seç
    random_indices = random.sample(range(len(df)), 14)
    manual_test_indices = [179]
    
    for idx in random_indices:
        # Gerçek veriyi al
        original_row = df.iloc[idx]
        
        # Test için hazırlanmış veriyi al (Scale edilmiş hali)
        # Not: X_scaled tüm veriyi içerir, idx ile eşleşen satırı bulmamız lazım.
        # Ancak train_test_split yaptığımız için X_scaled doğrudan df ile aynı sırada değil.
        # Bu yüzden basitlik adına, seçtiğimiz idx'in özelliklerini manuel scale edip tahmin yapacağız.
        
        features_raw = original_row[features_cols].values.reshape(1, -1)
        features_scaled = scaler_X.transform(features_raw)
        features_tensor = torch.FloatTensor(features_scaled)
        
        # Tahmin
        predicted_scaled = model(features_tensor).item()
        predicted_real = scaler_y.inverse_transform([[predicted_scaled]])[0][0]
        
        # Gerçek Değer
        actual_real = original_row['Tur_Zaman_Saniye']
        
        # Araç Adı
        car_name = original_row['Araç Modeli']
        
        print(f"Araç: {car_name}")
        print(f"Gerçek Süre: {actual_real:.2f} sn ({int(actual_real//60)}:{actual_real%60:05.2f})")
        print(f"Tahmini Süre: {predicted_real:.2f} sn ({int(predicted_real//60)}:{predicted_real%60:05.2f})")
        print(f"Hata: {abs(actual_real - predicted_real):.2f} sn")
        print("-" * 30)
    
    for idx in manual_test_indices:
        original_row = df.iloc[idx]
        features_raw = original_row[features_cols].values.reshape(1, -1)
        features_scaled = scaler_X.transform(features_raw)
        features_tensor = torch.FloatTensor(features_scaled)

        predicted_scaled = model(features_tensor).item()
        predicted_real = scaler_y.inverse_transform([[predicted_scaled]])[0][0]

        actual_real = original_row['Tur_Zaman_Saniye']
        car_name = original_row['Araç Modeli']

        print(f"Araç: {car_name}")
        print(f"Gerçek Süre: {actual_real:.2f} sn ({int(actual_real//60)}:{actual_real%60:05.2f})")
        print(f"Tahmini Süre: {predicted_real:.2f} sn ({int(predicted_real//60)}:{predicted_real%60:05.2f})")
        print(f"Hata: {abs(actual_real - predicted_real):.2f} sn")
        print("-" * 30)
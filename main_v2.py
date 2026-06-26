import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# 1. VERİ YÜKLEME
try:
    df = pd.read_excel('train_v7.xlsx')
    print("Veri başarıyla yüklendi. Satır sayısı:", len(df))
except FileNotFoundError:
    print("HATA: 'train_v7.xlsx' dosyası bulunamadı.")
    raise SystemExit

# 2. VERİ ÖN İŞLEME
def time_to_seconds(time_str):
    if isinstance(time_str, str):
        parts = time_str.split(':')
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    else:
        return time_str

df['Tur_Zaman_Saniye'] = df['Tur Zamanı'].apply(time_to_seconds)

# Özellikler ve Hedef
features_cols = ['Yıl', 'Motor', 'Güç (HP)', 'Tork (Nm)', 'Şanz.', 'Vites', 'Karoseri', 'Ağırlık', 'Çekiş', 'Aks Mesafesi (mm)', 'Ön Lastik (mm)', 'Arka Lastik (mm)', 'Yarış Arabası']
target_col = 'Tur_Zaman_Saniye'

X = df[features_cols].values
y = df[target_col].values.reshape(-1, 1)

# 3. MODEL MİMARİSİ
class NurburgringNet(nn.Module):
    def __init__(self, input_dim):
        super(NurburgringNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.15),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# 4. K-FOLD CROSS VALIDATION (k=5)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
criterion = nn.MSELoss()
epochs = 60

# Sonuçları saklamak için listeler
all_losses = []
all_test_errors = []

print("\n--- BAŞLIYOR: 5-KATLI ÇAPRAZ DOĞRULAMA (K-FOLD CV) ---")

for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    print(f"\n>>> FOLD {fold+1}/5")
    
    # Veriyi böl
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Ölçeklendirme (Her fold için yeniden fit edilir)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    
    # Tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    
    # DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Modeli sıfırdan oluştur
    input_dim = X_train.shape[1]
    model = NurburgringNet(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.00102)
    
    # Eğitim Döngüsü
    running_loss = 0.0
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f'  Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')
            
    all_losses.append(running_loss / len(train_loader))
    
    # Test Döngüsü (Tamir mekanizması kaldırıldı)
    print(f"  >>> FOLD {fold+1} TEST SÜRECİ")
    model.eval()
    fold_errors = []
    
    with torch.no_grad():
        for i in range(len(X_val)):
            features_tensor = X_val_tensor[i].unsqueeze(0) # Tek veri için boyut düzenlemesi
            
            actual_real = y_val[i][0]
            
            # Tahmin
            predicted_scaled = model(features_tensor).item()
            predicted_real = scaler_y.inverse_transform([[predicted_scaled]])[0][0]
            
            error = abs(actual_real - predicted_real)
            fold_errors.append(error)
            
    avg_fold_error = np.mean(fold_errors)
    all_test_errors.append(avg_fold_error)
    print(f"  >>> FOLD {fold+1} ORTALAMA HATA: {avg_fold_error:.2f} sn")

# 5. GENEL SONUÇLAR
print("\n--- 5-KATLI ÇAPRAZ DOĞRULAMA SONUÇLARI ---")
print(f"Ortalama Training Loss: {np.mean(all_losses):.4f}")
print(f"Ortalama Test Hatası: {np.mean(all_test_errors):.2f} sn")
print(f"Standart Sapma (Test Hatası): {np.std(all_test_errors):.2f} sn")
print("-" * 30)
print("NOT: Düşük standart sapma, modelin kararlı olduğunu gösterir.")
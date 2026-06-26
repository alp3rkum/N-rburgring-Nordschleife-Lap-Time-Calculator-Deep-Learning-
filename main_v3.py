import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def time_to_seconds(time_str):
    if isinstance(time_str, str):
        parts = time_str.split(':')
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    else:
        return time_str

def augment_data(df, num_augmentations=2):
    """
    Her araç için num_augmentations kadar sanal varyasyon üretir.
    """
    augmented_rows = []
    
    for index, row in df.iterrows():
        # Orijinal satırı koru
        augmented_rows.append(row)
        
        for i in range(num_augmentations):
            new_row = row.copy()
            
            # 1. Güç ve Ağırlığa küçük gürültü ekle
            hp_noise = np.random.uniform(-0.03, 0.03) # ±%3
            weight_noise = np.random.uniform(-0.02, 0.02) # ±%2
            
            new_row['Güç (HP)'] = row['Güç (HP)'] * (1 + hp_noise)
            new_row['Ağırlık'] = row['Ağırlık'] * (1 + weight_noise)
            
            # 2. Tur Zamanına gürültü ekle (Fiziksel ilişkiyi basitçe simüle et)
            # Güç artarsa süre düşer, ağırlık artarsa süre artar
            time_noise_factor = -0.5 * hp_noise + 0.3 * weight_noise 
            # Ekstra rastgelelik
            time_noise_factor += np.random.normal(0, 0.01) 
            
            # Tur zamanını saniyeye çevirip işlem yapalım
            current_time_sec = time_to_seconds(row['Tur Zamanı'])
            new_time_sec = current_time_sec * (1 + time_noise_factor)
            
            # Saniyeyi tekrar MM:SS.ss formatına çevir
            minutes = int(new_time_sec // 60)
            seconds = new_time_sec % 60
            new_row['Tur Zamanı'] = f"{minutes}:{seconds:05.2f}"
            
            # Araç ismine varyasyon belirtisi ekle (Opsiyonel, karışıklığı önlemek için)
            # new_row['Araç Modeli'] = f"{row['Araç Modeli']} (Aug {i+1})"
            
            augmented_rows.append(new_row)
            
    return pd.DataFrame(augmented_rows)

# --- YENİ EKLENEN KISIM: FEATURE FACTORY ---
def add_engineered_features(df):
    """
    Ham verilere fiziksel anlam katan türetilmiş özellikler ekler.
    """
    df = df.copy() # Orijinal dataframe'i korumak için kopya alıyoruz
    
    # Bölme işleminde sıfıra bölünme hatasını önlemek için küçük bir epsilon ekleyebiliriz
    # Ama verimizde ağırlık 0 olan araç olmadığı için direkt bölebiliriz.
    
    # 1. Güç/Ağırlık Oranı (Hızlanma Potansiyeli)
    df['HP_Weight'] = df['Güç (HP)'] / df['Ağırlık']
    
    # 2. Tork/Ağırlık Oranı (Viraj Çıkışı Performansı)
    df['Torque_Weight'] = df['Tork (Nm)'] / df['Ağırlık']
    
    # 3. Toplam Lastik Genişliği (Toplam Grip Potansiyeli)
    df['Total_Tyre_Width'] = df['Ön Lastik (mm)'] + df['Arka Lastik (mm)']
    
    # 4. Grip Verimliliği (Lastik Genişliği / Ağırlık)
    df['Grip_Efficiency'] = df['Total_Tyre_Width'] / df['Ağırlık']
    
    # 5. Aks Mesafesi Oranı (Denge/Stabilite İndikatörü)
    df['Wheelbase_Ratio'] = df['Aks Mesafesi (mm)'] / df['Ağırlık']
    
    return df

# 1. VERİ YÜKLEME
try:
    df = pd.read_excel('train_v7.xlsx')
    df = augment_data(df, 4)
    print("Veri başarıyla yüklendi. Satır sayısı:", len(df))
except FileNotFoundError:
    print("HATA: 'train_v7.xlsx' dosyası bulunamadı.")
    raise SystemExit

# --- YENİ EKLENEN KISIM: FEATURE HESAPLAMA ---
# Ham veriyi al, üzerine mühendislik özelliklerini ekle
df = add_engineered_features(df)
print("Mühendislik özellikleri eklendi. Yeni sütunlar:", ['HP_Weight', 'Torque_Weight', 'Total_Tyre_Width', 'Grip_Efficiency', 'Wheelbase_Ratio'])

# 2. VERİ ÖN İŞLEME



df['Tur_Zaman_Saniye'] = df['Tur Zamanı'].apply(time_to_seconds)

# Özellikler ve Hedef
# DİKKAT: Buraya yeni eklediğimiz sütunların isimlerini de ekliyoruz!
features_cols = [
    'Yıl', 'Motor', 'Güç (HP)', 'Tork (Nm)', 'Şanz.', 'Vites', 'Karoseri', 
    'Ağırlık', 'Çekiş', 'Aks Mesafesi (mm)', 'Ön Lastik (mm)', 'Arka Lastik (mm)', 
    'Yarış Arabası',
    # YENİ EKLENENLER:
    'HP_Weight', 'Torque_Weight', 'Total_Tyre_Width', 'Grip_Efficiency', 'Wheelbase_Ratio'
]

target_col = 'Tur_Zaman_Saniye'

X = df[features_cols].values
y = df[target_col].values.reshape(-1, 1)

# 3. MODEL MİMARİSİ
class NurburgringNet(nn.Module):
    def __init__(self, input_dim):
        super(NurburgringNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.15),
            nn.Linear(32, 16),
            nn.GELU(),
            #nn.Linear(16,8),
            #nn.GELU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# 4. K-FOLD CROSS VALIDATION (k=5)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
criterion = nn.MSELoss()
epochs = 60

all_losses = []
all_test_errors = []

print("\n--- 5-KATLI ÇAPRAZ DOĞRULAMA (K-FOLD CV) ---")

for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    print(f"\n>>> FOLD {fold+1}/5")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    input_dim = X_train.shape[1] # Artık 18 özellik var (13 eski + 5 yeni)
    model = NurburgringNet(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.00097) # LR'i standart 0.001'e çektim, daha stabil
    
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
    
    print(f"  >>> FOLD {fold+1} TEST SÜRECİ")
    model.eval()
    fold_errors = []
    
    with torch.no_grad():
        for i in range(len(X_val)):
            features_tensor = X_val_tensor[i].unsqueeze(0)
            actual_real = y_val[i][0]
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
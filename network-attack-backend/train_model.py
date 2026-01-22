import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import joblib

# Load your dataset
df = pd.read_csv('dataset.csv', on_bad_lines='skip')

# Find label column
possible_label_cols = [col for col in df.columns if 'attack' in col.lower() or 'label' in col.lower()]
label_column = possible_label_cols[0] if possible_label_cols else df.columns[-1]

# Attack mapping
attack_mapping = {
    'DDOS_Slowloris': 'denial of service',
    'DOS_SYN_Hping': 'denial of service',
    'Metasploit_Brute_Force_SSH': 'denial of service',
    'NMAP_FIN_SCAN': 'spoofing',
    'NMAP_OS_DETECTION': 'spoofing',
    'NMAP_TCP_scan': 'spoofing',
    'NMAP_UDP_SCAN': 'spoofing',
    'NMAP_XMAS_TREE_SCAN': 'spoofing',
    'MQTT_Publish': 'spoofing',
    'Thing_Speak': 'spoofing',
    'Wipro_bulb': 'spoofing',
    'ARP_poisioning': 'deauthentication',
}
df['project_attack_type'] = df[label_column].map(attack_mapping).fillna('other')

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols = [col for col in categorical_cols if col not in [label_column, 'project_attack_type']]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(df[col].astype(str))
    df[col] = le.transform(df[col].astype(str))
    label_encoders[col] = le

# Filter target attacks
target_attacks = ['jamming', 'spoofing', 'deauthentication', 'denial of service']
df = df[df['project_attack_type'].isin(target_attacks)].copy()

# Handle missing values
df = df.fillna(df.median(numeric_only=True))
df.dropna(subset=['project_attack_type'], inplace=True)

# Prepare features
X = df.drop([label_column, 'project_attack_type'], axis=1)
y = df['project_attack_type']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance classes
max_size = y.value_counts().max()
target_samples = {
    'denial of service': int(max_size * 0.50),
    'spoofing': int(max_size * 0.35),
    'deauthentication': int(max_size * 0.30)
}

balanced_X, balanced_y = [], []
for attack_type in y.unique():
    mask = (y == attack_type)
    X_class = X_scaled[mask]
    y_class = y[mask]
    if attack_type in target_samples:
        target = target_samples[attack_type]
        indices = resample(range(len(y_class)), n_samples=target, random_state=2, replace=(len(y_class) < target))
        balanced_X.append(X_class[indices])
        balanced_y.append(y_class.iloc[indices])

X_scaled = np.vstack(balanced_X)
y = pd.concat(balanced_y)

# Apply PCA
pca = PCA(n_components=0.95, random_state=2)
X_pca = pca.fit_transform(X_scaled)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=2, stratify=y)

# Train model
clf = RandomForestClassifier(
    n_estimators=100, 
    random_state=2, 
    class_weight='balanced',
    max_depth=15, 
    min_samples_split=10, 
    min_samples_leaf=5
)
clf.fit(X_train, y_train)

# Save everything
joblib.dump(clf, 'network_attack_model.joblib')
joblib.dump(scaler, 'network_attack_scaler.joblib')
joblib.dump(pca, 'network_attack_pca.joblib')
joblib.dump(label_encoders, 'network_attack_label_encoders.joblib')
np.save('X_train.npy', X_train)  # Save for LIME

print("✓ Models trained and saved successfully!")
print(f"✓ PCA components: {pca.n_components_}")
print(f"✓ Classes: {clf.classes_}")

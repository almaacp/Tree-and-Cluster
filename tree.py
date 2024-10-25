#%%
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from dtreeviz import model
from matplotlib import pyplot as plt

plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
plt.rcParams.update({'font.size': 14})

#%%
# Membaca dataset
syndrome = pd.read_csv("Metabolic Syndrome.csv")

# Mendeteksi dan menghapus NaN
print("Jumlah NaN sebelum penghapusan:")
print(syndrome.isna().sum())
syndrome = syndrome.dropna()
print("\nJumlah NaN setelah penghapusan:")
print(syndrome.isna().sum())

#%%
# Memisahkan fitur dan target
X = syndrome.drop(['MetabolicSyndrome', 'seqn'], axis=1)
y = syndrome['MetabolicSyndrome']
X = pd.get_dummies(X, drop_first=True)

#%%
# Melatih model RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X, y)

#%%
# Visualisasi pohon keputusan
plt.figure(figsize=(20, 20))
_ = tree.plot_tree(rf.estimators_[0], feature_names=X.columns, filled=True)

#%%
# Memeriksa kedalaman pohon
print(rf.estimators_[0].tree_.max_depth)

#%%
# Mengatur ulang model dengan max_depth=3
rf = RandomForestRegressor(n_estimators=100, max_depth=3)
rf.fit(X, y)

#%%
# Visualisasi pohon keputusan yang baru
plt.figure(figsize=(20, 20))
_ = tree.plot_tree(rf.estimators_[0], feature_names=X.columns, filled=True)

#%%
# Visualisasi pohon keputusan dengan dtreeviz
viz = model(rf.estimators_[0], X, y, feature_names=X.columns, target_name='MetabolicSyndrome')
viz.view()

# %%

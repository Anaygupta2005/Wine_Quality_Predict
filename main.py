import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt

wineData = pd.read_csv("wineQualityReds.csv")

wineData.drop("Wine",axis=1,inplace=True)

wineQuality = wineData["quality"]

wineData.drop("quality",axis=1,inplace=True)

print(wineData)
print(wineQuality)

norm = Normalizer()
wineData_norm = pd.DataFrame(norm.transform(wineData),columns=wineData.columns)

print(wineData_norm)

ks = range(1,11)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(wineData_norm)
    inertias.append(model.inertia_)

plt.plot(ks, inertias, "-o")
plt.xlabel("Number of Clusters, k")
plt.ylabel("Inertia")
plt.show()

model = KMeans(n_clusters=6, random_state=2023)
model.fit(wineData_norm)
labels = model.predict(wineData_norm)
wineData_norm["Cluster"] = pd.Series(labels)
print(wineData_norm)

wineData_norm["quality"] = wineQuality
crosstab = pd.crosstab(wineData_norm["quality"], wineData_norm["Cluster"])
print(crosstab)

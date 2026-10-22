from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
customer = [(1000,2),
            (1500,3),
            (2000,2),
            (5000,6),
            (6000,7),
            (5500,5),
            (12000,12),
            (15000,14),
            (13000,11),
            (3000,4)
]
scaler =StandardScaler()
data_scaled = scaler.fit_transform(customer)
model = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = model.fit_predict(data_scaled)
for i in range(10):
    print(f"customer{i+1} belongs to cluster {clusters[i]}")
    print(f"cluster 0 total: {list(clusters).count(0)}")
    print(f"cluster 1 total: {list(clusters).count(1)}")
    print(f"cluster 2 total: {list(clusters).count(2)}")

import matplotlib.pyplot as plt
spending= [x[0] for x in customer ]
visits= [x[1] for x in customer]
plt.scatter(spending, visits, c=clusters, cmap='viridis', s=100)
plt.xlabel("Monthly Spending (KES)")
plt.ylabel("Monthly Visits")
plt.title("customer groups")
plt.show()

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('seaborn-poster')
style.use('ggplot')
import seaborn as sns
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import requests
# Downloading the csv file(The Kaggle one) from a link to use it
file_url = "https://drive.google.com/uc?export=download&id=1rUxETGW4GJtXCtGgsIcQdchGLafkmq8H"
req = requests.get(file_url)
csv_file = open('customer_dataset.csv', 'wb')
csv_file.write(req.content)
csv_file.close()
display(df.describe())
display(df.info())
df = df.rename(columns={
    "Annual Income (k$)": "An. Income",
    "Spending Score (1-100)": "Spend Score"
})
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(ax=ax, data=df.isnull())
# Distribution of Income, Spending score, Gender, Age
fig, axes = plt.subplots(2, 2, figsize=(20, 10))
sns.set(style='darkgrid')

sns.histplot(df['Age'], ax=axes[0][0], kde=True, color='crimson')
axes[0][0].set_title('Distribution of Age')
sns.histplot(df['Spend Score'], ax=axes[0][1], kde=True, color='blue')
axes[0][1].set_title('Distribution of Spending Score')
sns.histplot(df['An. Income'], ax=axes[1][0], kde=True, color='coral')
axes[1][0].set_title('Distribution of Annual Income')

colors = sns.color_palette('pastel')[0:5]
axes[1][1].pie(df['Gender'].value_counts(), colors = colors, labels=['Female', 'Male'], autopct='%.0f%%')
axes[1][1].set_title('Distribution Of Genders in Customers')
fig.tight_layout(pad=3.0)
plt.savefig("./dist_age_sc_an.png", bbox_inches="tight")
# Preprocessing the data
one_hot = pd.get_dummies(df['Gender'])
df_encoded = df.join(one_hot)
df_encoded.drop('Gender', axis=1, inplace=True)
df_encoded.head()
# Correlation
df_new = df_encoded.drop(['CustomerID'], axis=1)
corr = df_new.corr()
print(corr['Age']['Age'])

columns = df_new.columns
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
c_ax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(c_ax)
ticks = np.arange(0,len(columns),1)
ax.set_xticks(ticks)
plt.title('Correlation Between Different Features', y=-0.2)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(columns)
ax.set_yticklabels(columns)

columns
for i in range(len(columns)):
    for j in range(len(columns)):
        text = ax.text(j, i, 
                       "{:.2f}".format(corr[columns[i]][columns[j]]),
                       ha="center", va="center", color="b")

plt.savefig("./corr.png", bbox_inches="tight")
plt.show()
male_spending = df_encoded[df_encoded['Male'] == 1]['Spend Score']
female_spending = df_encoded[df_encoded['Female'] == 1]['Spend Score']
fig, ax = plt.subplots(1, 2, figsize=(15, 8))
spend_columns = [male_spending, female_spending]

ax[0].boxplot(spend_columns)
ax[0].set_xticklabels(["Male", "Female"])
ax[0].set_title('Spending Range for Genders')
ax[0].set_ylabel("Spending Score")

male_age = df_encoded[df_encoded['Male'] == 1]['Age']
female_age = df_encoded[df_encoded['Female'] == 1]['Age']
age_columns = [male_age, female_age]
ax[1].boxplot(age_columns)
ax[1].set_xticklabels(["Male", "Female"])
ax[1].set_title('Age Range for Genders')
ax[1].set_ylabel("Age (in years)")
plt.savefig("./box_plots.png", bbox_inches="tight")

plt.show()

# medians = [round(item.get_ydata()[0], 1) for item in bp['medians']]
fig = plt.figure(figsize=(20, 5))

spending_age_male = df_encoded[df_encoded['Male'] == 1].loc[:, ['Age', 'Spend Score']]
spending_age_male = spending_age_male.sort_values('Age')
sns.lineplot(spending_age_male['Age'], spending_age_male['Spend Score'], label='Male')

spending_age_female = df_encoded[df_encoded['Female'] == 1].loc[:, ['Age', 'Spend Score']]
spending_age_female = spending_age_female.sort_values('Age')
sns.lineplot(spending_age_female['Age'], spending_age_female['Spend Score'], label='Female')

plt.xlabel("Age (in years)")
plt.ylabel("Spending Score")
plt.title("Age vs Spending Score")
plt.legend()
plt.savefig("./spend_sc_vs_age.png", bbox_inches="tight")
plt.show()
fig = plt.figure(figsize=(20, 5))

spending_age_male = df_encoded[df_encoded['Male'] == 1].loc[:, ['Age', 'An. Income']]
spending_age_male = spending_age_male.sort_values('Age')
sns.lineplot(spending_age_male['Age'], spending_age_male['An. Income'], label='Male')

spending_age_female = df_encoded[df_encoded['Female'] == 1].loc[:, ['Age', 'An. Income']]
spending_age_female = spending_age_female.sort_values('Age')
sns.lineplot(spending_age_female['Age'], spending_age_female['An. Income'], label='Female')

plt.xlabel("Age (in years)")
plt.ylabel("Annual Income")
plt.title("Age vs Annual Income")
plt.legend()
plt.savefig("./an_vs_age.png", bbox_inches="tight")
plt.show()
income_spend = df.loc[:, ["An. Income", "Spend Score"]].sort_values("An. Income")
sns.lineplot(income_spend["An. Income"], income_spend["Spend Score"])
pax = pd.plotting.scatter_matrix(df.drop("CustomerID", axis=1), figsize=(10,10), marker = 'o', hist_kwds = {'bins': 50}, s = 60, alpha = 0.8)
plt.show()
X=df_encoded.drop(["CustomerID", "Male", "Female"], axis=1)
scaler = MinMaxScaler()
scaler.fit(X)
X=scaler.transform(X)
inertia = []
for i in range(1,11):
    kmeans = KMeans(
        n_clusters=i, init="k-means++",
        max_iter=300,
        n_init=10,
        tol=1e-04, random_state=42
    )
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6, 6))
plt.plot(range(1,11), inertia)
plt.title("Elbow Method")
plt.xlabel("No. Of Clusters")
plt.ylabel("Inertia")
plt.savefig("three_params_elbow.png", bbox_inches="tight")
plt.show()
model = KMeans(n_clusters = 5, init = "k-means++", max_iter = 300, n_init = 10, random_state = 42)
y_clusters = model.fit_predict(X)

fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(y_clusters, ax=ax)
ax.set_xlabel("Cluster No.")
plt.savefig("./three_params_cluster_count.png", bbox_inches="tight")
fig = plt.figure(figsize = (12,12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[y_clusters == 0,0],X[y_clusters == 0,1],X[y_clusters == 0,2], s = 40 , color = 'blue', label = "Cluster 0")
ax.scatter(X[y_clusters == 1,0],X[y_clusters == 1,1],X[y_clusters == 1,2], s = 40 , color = 'orange', label = "Cluster 1")
ax.scatter(X[y_clusters == 2,0],X[y_clusters == 2,1],X[y_clusters == 2,2], s = 40 , color = 'green', label = "Cluster 2")
ax.scatter(X[y_clusters == 3,0],X[y_clusters == 3,1],X[y_clusters == 3,2], s = 40 , color = 'coral', label = "Cluster 3")
ax.scatter(X[y_clusters == 4,0],X[y_clusters == 4,1],X[y_clusters == 4,2], s = 40 , color = 'purple', label = "Cluster 4")
ax.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:, 1], model.cluster_centers_[:, 2], s = 50, c = 'black' , label = 'centeroid')
ax.set_xlabel('Age (in years)', labelpad=25)
ax.set_ylabel('Annual Income (in K$)', labelpad=25)
ax.set_zlabel('Spending Score', labelpad=20)
ax.legend()
plt.savefig("3d_plot.png", bbox_inches="tight")
plt.show()
X2=df_encoded.drop(["CustomerID", "Male", "Female", "Age"], axis=1)
scaler = MinMaxScaler()
scaler.fit(X2)
X2=scaler.transform(X2)
inertia2 = []
for i in range(1,11):
    kmeans = KMeans(
        n_clusters=i, init="k-means++",
        max_iter=200,
        n_init=10,
        tol=1e-04, random_state=42
    )
    kmeans.fit(X2)
    inertia2.append(kmeans.inertia_)

plt.figure(figsize=(6, 6))
plt.plot(range(1,11), inertia2)
plt.title("Elbow Method")
plt.xlabel("No. Of Clusters")
plt.ylabel("Inertia")
plt.show()
model2 = KMeans(n_clusters = 5, init = "k-means++", max_iter = 300, n_init = 10, random_state = 42)
y_clusters2 = model2.fit_predict(X2)

fig, ax = plt.subplots(1, 2, figsize=(20, 8))
sns.countplot(y_clusters2, ax=ax[0])
ax[0].set_xlabel("Cluster No.") 

X2 = np.array(X2)
# ax = fig.add_subplot(111, projection='3d')
ax[1].scatter(X2[y_clusters2 == 0,0], X2[y_clusters2 == 0,1], s = 40 , color = 'blue', label = "Cluster 0 - General")
ax[1].scatter(X2[y_clusters2 == 1,0], X2[y_clusters2 == 1,1], s = 40 , color = 'orange', label = "Cluster 1 - Miser")
ax[1].scatter(X2[y_clusters2 == 2,0], X2[y_clusters2 == 2,1], s = 40 , color = 'green', label = "Cluster 2 - Careful")
ax[1].scatter(X2[y_clusters2 == 3,0], X2[y_clusters2 == 3,1], s = 40 , color = 'coral', label = "Cluster 3 - Spendthrift")
ax[1].scatter(X2[y_clusters2 == 4,0], X2[y_clusters2 == 4,1], s = 40 , color = 'purple', label = "Cluster 4 - Important")
ax[1].scatter(model2.cluster_centers_[:,0], model2.cluster_centers_[:, 1], s = 50, c = 'black' , label = 'Centeroid')
ax[1].set_xlabel('Annual Income (in K$)', labelpad=25)
ax[1].set_ylabel('Spending Score', labelpad=20)
ax[1].legend()
plt.savefig("./two_params_count_plot.png", bbox_inches="tight")
plt.show()

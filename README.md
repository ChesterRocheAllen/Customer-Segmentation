# Customer-Segmentation
Clustering algorithms based on various customer attributes (e.g., Age, Income, Spending Score, etc.) and visualizes the results through multiple plots. data from httpswww.kaggle.comdatasetsdatascientistannacustomers-dataset
## K-Means clustering algorithm
### Python code structure 
### Importing Required Libraries
pandas: Used for data manipulation and analysis.

**matplotlib.pyplot and seaborn:** For data visualization.

**LabelEncoder:** Converts categorical labels (like gender) into numeric values.

**StandardScaler:** Scales numerical features to have a mean of 0 and standard deviation of 1.

**KMeans:** The K-Means clustering algorithm, used for unsupervised learning.
### Data Handling
**df = pd.read_csv('Customers.csv')**   --> Gets the CSV file containing customer data.

**le = LabelEncoder() 
df['Gender'] = le.fit_transform(df['Gender'])**  --> LabelEncoder is used to convert the Gender column into binary. 

**df = pd.get_dummies(df, columns=['Profession'], drop_first=True)** -->One-Hot Encoding is applied to the Profession column. This creates a binary column for each profession.

**numeric_features = ['Age', 'Annual Income ($)', 'Spending Score (1-100)', 'Work Experience', 'Family Size']** --> Numerical columns of the dataset chosen for scaling.

**scaler = StandardScaler()**  

**scaled_data = scaler.fit_transform(df[numeric_features])** --> StandardScaler is used to scale the numeric features  making the K-Means algorithm more effective
### Elbow Method 
**wcss = []**

**for i in range(1, 11):** --> Model for a range of cluster numbers (1 to 10)

   **kmeans = KMeans(n_clusters=i, random_state=42)**
   
   **kmeans.fit(scaled_data)**
    
   **wcss.append(kmeans.inertia_)** -->measures the compactness of clusters 
###  Applying K-Means Clustering
**kmeans = KMeans(n_clusters=4, random_state=42)** --> K-Means is applied with n_clusters=4 to assign each customer to one of the 4 clusters.

**df['Cluster'] = kmeans.fit_predict(scaled_data)** --> The cluster labels are stored in a new column called Cluster in the DataFrame.
### Inverse Scaling
**original_values = scaler.inverse_transform(scaled_data)** --> Scaled data is converted back to its original scale for easier interpretation in visualizations.

**df[numeric_features] = original_values**
## Results 
### Clustering by Age, Income, Spending, Work Expirence and  Family Size

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from function import label_facies,accuracy, accuracy_adjacent ,make_facies_log_plot, compare_facies_plot1, compare_facies_plot2
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

facies_colors = ['#F4D03F', '#F5B041', '#DC7633', '#6E2C00',
                 '#1B4F72', '#2E86C1', '#AED6F1', '#A569BD', '#196F3D']
facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D','PS', 'BS']
facies_color_map = {}
for ind, label in enumerate(facies_labels):
    facies_color_map[label] = facies_colors[ind]
adjacent_facies = np.array([[1], [0, 2], [1], [4], [3, 5], [4, 6, 7], [5, 7], [5, 6, 8], [6, 7]], dtype=object)
facies_names_abbreviation = {
    'SS': 'Nonmarine sandstone',
    'CSiS': 'Nonmarine coarse siltstone',
    'FSiS': 'Nonmarine fine siltstone',
    'SiSh': 'Marine siltstone and shale',
    'MS': 'Mudstone (limestone)',
    'WS': 'Wackestone (limestone)',
    'D': 'Dolomite',
    'PS': 'Packstone-grainstone (limestone)',
    'BS': 'Phylloid-algal bafflestone (limestone)'
}

data = pd.read_csv("train.csv")
data.loc[:, 'FaciesLabels'] = data.apply(lambda row: label_facies(row, facies_labels), axis=1)
training_data = data.copy()
training_data.dropna(inplace=True)

x = training_data.drop(["Depth" , "DeltaPHI" , "FaciesLabels" ,"Formation", "RELPOS", "PE", "Well Name"], axis=1)
y = training_data.PE
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


model_s = RandomForestRegressor(
    random_state=6,
    criterion='absolute_error',
    max_depth=18,
    max_features='log2',
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=200
)
model_s.fit(x_scaled, y)

data_prediction = data.copy()
well_features = data_prediction[x.columns]
scaler = preprocessing.StandardScaler().fit(well_features)
X_data_scaled = scaler.transform(well_features)

# Make predictions using randomforest
data_prediction["Synthesis_PE"] = model_s.predict(X_data_scaled)
data_prediction["PE"] = np.where(data["PE"].isnull(), data_prediction["Synthesis_PE"], data_prediction["PE"])
final_processed_data = data_prediction.drop("Synthesis_PE", axis=1)


# feature selection after preprosessing
y_data = final_processed_data['Facies'].values
final_well_features = final_processed_data[
    ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']]

scalar = StandardScaler()
X_final_scaled = scaler.fit_transform(final_well_features)

# KNN model buildup
model_knn = KNeighborsClassifier(algorithm ='ball_tree', leaf_size = 18, n_neighbors = 4)
model_knn.fit(X_final_scaled,y_data)

# SVM model buildup
model_svm = svm.SVC(C=10, gamma=1)
model_svm.fit(X_final_scaled,y_data)

# Logistic Regression model
model_logi = LogisticRegression(verbose=1 ,n_jobs=4 ,solver='newton-cg')
model_logi.fit(X_final_scaled,y_data)


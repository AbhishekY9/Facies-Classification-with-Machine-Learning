
**Facies Classification with Machine Learning**

**Project Overview**

This project aims to classify lithological facies from well log data using machine learning techniques. The identification of facies is crucial for petroleum engineers to understand hydrocarbon (HC) accumulation and for geologists to recognize geological activities below the subsurface.

**Data Description**

The dataset includes wireline log curves and geological constraining variables from multiple wells. The features used in this project are:

Gamma-ray (GR)

Resistivity logging (ILD_log10)

Photoelectric effect (PE)

Neutron-density porosity difference (Delta PHI)

Average neutron-density porosity (PHIND)

Additionally, two geological constraining variables are used:

Nonmarine-marine indicator (NM_M)

Relative position (RELPOS)

The target variable is the discrete facies classification, which consists of nine different rock types:

Nonmarine sandstone, 
Nonmarine coarse siltstone, 
Nonmarine fine siltstone, 
Marine siltstone and shale, 
Mudstone (limestone), 
Wackestone (limestone), 
Dolomite, 
Packstone-grainstone (limestone), 
Phylloid-algal bafflestone (limestone), 

**Methodology**

Data Collection: Gather log data from multiple wells, each labeled with a facies type based on core observations.

Data Preprocessing: Handle missing values, scale data, and split it into training and test sets.

Feature Engineering: Use wireline log curves and geological constraining variables as features for classification.

Model Development: Train and evaluate various machine learning algorithms, including:

Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Model Evaluation: Assess model performance using confusion matrices and F1 scores.

**Exploratory Data Analysis (EDA)**

EDA was performed to address missing values by generating synthetic PE log data using random forest regression. The data was then analyzed to understand patterns and prepare it for modeling.

**Algorithms Applied**

Logistic Regression: A fundamental classification algorithm used for binary and multi-class classification tasks.
K-Nearest Neighbors (KNN): A simple, instance-based learning algorithm that classifies data based on the closest training examples in the feature space.
Support Vector Machine (SVM): A powerful classification algorithm that finds the optimal hyperplane to separate different classes.
Results
The performance of each algorithm was evaluated using F1 scores and confusion matrices, both for exact and adjusted facies classifications:

Exact Facies Classification

SVM: F1 Score = 0.7148, 
Logistic Regression: F1 Score = 0.5855, 
KNN: F1 Score = 0.7280

Adjusted Facies Classification

SVM: F1 Score = 0.9269, 
Logistic Regression: F1 Score = 0.9528, 
KNN: F1 Score = 0.9359
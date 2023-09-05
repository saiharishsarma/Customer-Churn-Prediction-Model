# Customer-Churn-Prediction-Model

# Importing necessary libraries and tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from xgboost import plot_importance

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, roc_auc_score, f1_score, recall_score


import warnings
warnings.filterwarnings("ignore")
file = 'customer_churn_large_dataset.xlsx'
# Reading the excel dataset to pandas dataframe using following code by inserting file path of the dataset.
data = pd.read_excel(file)

![image](https://github.com/saiharishsarma/Customer-Churn-Prediction-Model/assets/99181593/2686bf18-8831-44fb-845c-c7bc2f79a1de)
![image](https://github.com/saiharishsarma/Customer-Churn-Prediction-Model/assets/99181593/457bb7ba-c4ee-4888-923c-b9238b0eb185)
![image](https://github.com/saiharishsarma/Customer-Churn-Prediction-Model/assets/99181593/3e58ac18-9889-4e42-80d2-de048a1b37e9)

# Data Preprocessing
# Modeling, Hyper parameter Tuning, Evaluation
# Modeling using Pycaret Library
# Model Application Interface Developement 
# Conclusion:
1. In Entire Dataset, there is no missing, duplicates and outlier items.
2. And also found there is no correlation among features, MinMax Scaler is used for Input Data Transformation.
3. Among All 6 Models in Individual Modeling, Almost All Models are giving test accuracy about 50%. It means, there is a need of more relavant features and data to capture the patterns from the data.
4. In Addition to this, Also tried with Pycaret Library. And Finally, Navie Bayes Model works some what well.
5. Eventhough, it is not that much accurate, went for automated deployment phase. 
6. Model can be developed more, if more features and details are available.

Thanks for Reading.

Feel Free to share your opinion/suggestion on this through mail.

Mail: saiharish.swarna@gmail.com




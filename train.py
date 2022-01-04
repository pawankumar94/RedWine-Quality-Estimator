# Imports
%matplotlib inline
import numpy as np
import scipy as sp
import pandas as pd
import sklearn
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from imblearn.over_sampling  import SMOTE
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn import metrics
from tabulate import tabulate
from sklearn import model_selection
seed = 42


def load_dataset(url="https://raw.githubusercontent.com/pawankumar94/RedWine-Quality-Estimator/main/winequality-red.csv"):
  df = pd.read_csv(url)
  return df

def preprocess_dataset(df):
  df["class"] = [1 if x >= 7 else 0 for x in df["quality"]]  # Feature Selection
  features = df.drop(["quality", "class"], axis = 1)  # Feature Extraction
  target = df["class"]
  return features, target

def plot_viz(df,features, target):
   correlation = df.corr()['quality'].drop("quality") # Correlation with  Target Variables
   
   plt.figure(figsize=(8,5))
   sns.countplot(x = 'quality', data = df)
   plt.xlabel('Ratings of wines')
   plt.ylabel('Amount')
   plt.title('Distribution of wine ratings')
   plt.savefig("Ditribution.png", dpi=150)
   plt.close()

   plt.figure(figsize=(8, 5))
   sns.heatmap(df.corr(), cbar = True, square = True, fmt = '.1f', annot = True, annot_kws={'size':8}, cmap="RdBu_r")
   plt.title("Correlation Matrix")
   plt.savefig("correlation.png", dpi=150)
   plt.tight_layout()
   plt.close()

   plt.figure(figsize=(8, 5))
   sns.barplot(x = ["Bad Quality Wine", "Good Quality Wine"], y = target.value_counts())
   plt.xlabel("classes")
   plt.ylabel("Counts")
   plt.title("Distribution of Good & Bad Wine Ratings Before Over-Sampling")
   plt.savefig("before-oversample.png", dpi=150)
   plt.tight_layout()
   plt.close()

   df.drop(columns=["class"], inplace = True)
   plot = df.corrwith(df.quality).plot.bar(figsize = (14, 5), title = "Correlation with quality", fontsize = 15, rot = 45, grid = True)
   fig = plot.get_figure()
   fig.savefig("Correlation-quality.png", dpi=150)

def classify1(clf, X, y, model_name, plot_dist=False):
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle= True, random_state=seed)
  oversample = SMOTE(k_neighbors=4)  # Used for Oversampling 
  new_X_train, new_y_train = oversample.fit_resample(X_train, y_train)

  if plot_dist:
    plt.figure(figsize=(8,5))
    sns.barplot(x = ["Bad Quality Wine", "Good Quality Wine"], y = new_y_train.value_counts())
    plt.xlabel("classes")
    plt.ylabel("Counts")
    plt.title("Distribution of Good & Bad Wine Ratings After Over-Sampling")
    plt.tight_layout()
    plt.savefig("After-oversample.png", dpi=150)
    plt.close()

  clf.fit(new_X_train, new_y_train)
  print("Accuracy on training set of:", model_name)
  print(clf.score(new_X_train, new_y_train)*100)
  print("Accuracy on testing set of:", model_name)
  print(clf.score(X_test, y_test)*100)
  y_pred = clf.predict(X_test)
  print("Classification Report of :", model_name)
  print(metrics.classification_report(y_test, y_pred))
  
  cm = metrics.confusion_matrix(y_test, y_pred)
  plt.figure(figsize=(10,5))
  disp = ConfusionMatrixDisplay(cm)
  disp.plot()
  plt.title("Confusion Matrix")
  plt.tight_layout()
  plt.savefig(model_name+"confusion.png",dpi=150)
  
  Training_score = clf.score(new_X_train, new_y_train)*100
  Testing_score = clf.score(X_test, y_test)*100
  sample_dict={}
  sample_dict[model_name] = [Training_score, Testing_score]

  return sample_dict

df = load_dataset()
imbalanced_features, imbalanced_target = preprocess_dataset(df)  # Features & target before oversampling
#plot_viz(df, imbalanced_features, imbalanced_target)  # used for EDA

# Instantiating Models
model_lr = LogisticRegression()
model_dt = DecisionTreeClassifier()
model_rf = RandomForestClassifier(max_depth=12, random_state = 42)

# Training of Models
results=[]
results.append(classify1(model_lr, imbalanced_features, imbalanced_target, "Logistic_Regression", True))
results.append(classify1(model_dt, imbalanced_features, imbalanced_target, "Decsion_Tree"))
results.append(classify1(model_rf, imbalanced_features, imbalanced_target, "Random_Forest"))

# used for Printing Generalization of Selected Models
vals = []
_ = [vals.append([list(each.keys())[0]] + list(each.values())[0]) for each in results]

tab = tabulate(vals, headers=["Name", "Training Score", "Testing Score"])
with open("Results.txt", "w") as fw:
  fw.write(tab)

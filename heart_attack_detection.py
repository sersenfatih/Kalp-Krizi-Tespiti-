import pandas as pd,pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sb
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



df = pd.read_csv("heart.csv")

df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure',
              'cholesterol', 'fasting_blood_sugar', 'rest_ecg',
              'max_heart_rate_achieved', 'exercise_induced angina',
              'st_depression', 'st_slope', 'num_major_vessels',
              'thalassemia', 'target']

f, ax = plt.subplots(figsize=(10,10))
sb.heatmap(df.corr(), annot=True, linewidth=0.5, fmt='.1f',ax=ax)
plt.show()

x = df.iloc[:,0:13].values
y = df.iloc[:,13].values


sc = StandardScaler()
sc_x = sc.fit_transform(x)

pca = PCA(n_components=13)

pca.fit(x)

def showVarianceRatio(pca):
    
    var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    plt.plot(var1)

showVarianceRatio(pca)

pca2 = PCA(n_components=3)
pca2.fit(x)
X1=pca2.fit_transform(x)

fig, axes = plt.subplots(1,2)
axes[0].scatter(x[:,0], x[:,12], c=y)
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[0].set_title('Before PCA')

axes[1].scatter(X1[:,0], X1[:,2], c=y)
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('After PCA')
plt.show()    

X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.2, random_state = 42)
model = RandomForestClassifier(max_depth=5)
model.fit(X_train, y_train)

y_predict = model.predict(X_test)


cm = confusion_matrix(y_test, y_predict)
sb.heatmap(cm/np.sum(cm), annot=True, fmt=".2%",cmap='Blues')
plt.show()

cm = confusion_matrix(y_test, y_predict)
sb.heatmap(cm, annot=True,cmap='Blues')
plt.show()

ac = accuracy_score(y_test, y_predict)
print("Accuracy is: {}".format(ac*100))

sensitivity = cm[1,1]/(cm[1,1]+cm[0,1])
print('Sensitivity : ', sensitivity )

specificity = cm[0,0]/(cm[1,0]+cm[0,0])
print('Specificity : ', specificity)



y_pred_quant = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)

fig, ax = plt.subplots()
ax.plot(fpr,tpr)
ax.plot([0,1], [0,1], transform = ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

auc(fpr, tpr)
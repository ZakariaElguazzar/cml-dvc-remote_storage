## main
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from imblearn.over_sampling import SMOTE
import joblib

from PIL import Image

## sklearn -- preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector

## sklearn -- models
from sklearn.ensemble import RandomForestClassifier

## sklearn -- metrics
from sklearn.metrics import f1_score, confusion_matrix


## --------------------- Data Preparation ---------------------------- ##

TRAIN_PATH = os.path.join(os.getcwd(), './data/dataset.csv')
df = pd.read_csv(TRAIN_PATH)

df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
df.drop(index=df[df['Age'] > 80].index.tolist(), axis=0, inplace=True)

X = df.drop(columns=['Exited'], axis=1)
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=45, stratify=y
)


## --------------------- Data Processing ---------------------------- ##

num_cols = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']
categ_cols = ['Gender', 'Geography']
ready_cols = list(set(X_train.columns) - set(num_cols) - set(categ_cols))

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_cols)),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categ_pipeline = Pipeline([
    ('selector', DataFrameSelector(categ_cols)),
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(drop='first', sparse_output=False))
])

ready_pipeline = Pipeline([
    ('selector', DataFrameSelector(ready_cols)),
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

all_pipeline = FeatureUnion([
    ('numerical', num_pipeline),
    ('categorical', categ_pipeline),
    ('ready', ready_pipeline)
])

X_train_final = all_pipeline.fit_transform(X_train)
X_test_final = all_pipeline.transform(X_test)


## --------------------- Imbalance Handling ---------------------------- ##

vals_count = 1 - (np.bincount(y_train) / len(y_train))
vals_count = vals_count / np.sum(vals_count)

dict_weights = {i: vals_count[i] for i in range(2)}

over = SMOTE(sampling_strategy=0.7)
X_train_resmapled, y_train_resampled = over.fit_resample(X_train_final, y_train)


## --------------------- Modeling ---------------------------- ##

with open('metrics.txt', 'w') as f:
    pass

os.makedirs('models', exist_ok=True)


def train_model(X_train, y_train, plot_name='', class_weight=None):
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        random_state=45,
        class_weight=class_weight
    )

    clf.fit(X_train, y_train)

    model_path = os.path.join('models', f"RandomForest_{plot_name}.joblib")
    joblib.dump(clf, model_path)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test_final)

    score_train = f1_score(y_train, y_pred_train)
    score_test = f1_score(y_test, y_pred_test)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, cbar=False, fmt='.2f', cmap='Blues')
    plt.title(f'{plot_name}')
    plt.xticks([0.5, 1.5], [False, True])
    plt.yticks([0.5, 1.5], [False, True])

    plt.savefig(f'{plot_name}.png', bbox_inches='tight', dpi=300)
    plt.close()

    with open('metrics.txt', 'a') as f:
        f.write(f'RandomForest {plot_name}\n')
        f.write(f"F1-score Training: {score_train*100:.2f}%\n")
        f.write(f"F1-score Validation: {score_test*100:.2f}%\n")
        f.write('----'*10 + '\n')

    return True


train_model(X_train_final, y_train, 'without-imbalance', None)
train_model(X_train_final, y_train, 'with-class-weights', dict_weights)
train_model(X_train_resmapled, y_train_resampled, 'with-SMOTE', None)

confusion_matrix_paths = [
    './without-imbalance.png',
    './with-class-weights.png',
    './with-SMOTE.png'
]

plt.figure(figsize=(15, 5))
for i, path in enumerate(confusion_matrix_paths, 1):
    img = Image.open(path)
    plt.subplot(1, 3, i)
    plt.imshow(img)
    plt.axis('off')

plt.suptitle('RandomForest', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('conf_matrix.png', bbox_inches='tight', dpi=300)

for path in confusion_matrix_paths:
    os.remove(path)


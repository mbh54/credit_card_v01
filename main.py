import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
import joblib

with open('data/sample_train_data.pkl', 'rb') as f:
    data = pickle.load(f)

with open('data/sample_train_label.pkl', 'rb') as f:
    data_label = pickle.load(f)

null_threshold = 0.8
null_percentages = data.isnull().sum() / len(data)
columns_to_drop = null_percentages[null_percentages > null_threshold].index.tolist()
data = data.drop(columns_to_drop, axis=1)

categorical_columns = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_68']
numeric_columns = [col for col in data.columns if col not in categorical_columns and col not in data.columns[:2]]

Impute_features = ColumnTransformer(
    [
    ('categorical', SimpleImputer(missing_values=np.nan, strategy='most_frequent'), categorical_columns),
    ('numeric', SimpleImputer(missing_values=np.nan, strategy='mean'), numeric_columns)
    ],
    remainder='passthrough')

est = Pipeline([ 
    ('Imput_features', Impute_features)
])

column_names = categorical_columns + numeric_columns + ['customer_ID', 'S_2']
data_fit = est.fit_transform(data)
data_imputed = pd.DataFrame(data_fit, columns=column_names).reset_index(drop='index')

data_grouped = data_imputed.loc[:,~data_imputed.columns.isin(['S_2', 'D_63', 'D_64'])].groupby('customer_ID').mean()

data_random_order, label_random_order = shuffle(data_grouped, data_label, random_state=42)

est_logistic = LogisticRegression(max_iter=2000)
cv = 10
n_jobs = 2
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
param_grid = dict(solver=solvers,penalty=penalty,C=c_values)

gs_logistic = GridSearchCV(
    est_logistic,
    param_grid = param_grid,  
    cv=cv, 
    n_jobs=n_jobs, 
)
gs_logistic_model = gs_logistic.fit(data_random_order, label_random_order.iloc[:,1]);

print("Best: %f using %s" % (gs_logistic_model.best_score_, gs_logistic_model.best_params_))
means = gs_logistic_model.cv_results_['mean_test_score']
stds = gs_logistic_model.cv_results_['std_test_score']
params = gs_logistic_model.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

joblib.dump(gs_logistic_model, 'pre_trained_model.joblib')
print('Model was successfully saved.')


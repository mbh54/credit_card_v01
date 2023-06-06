######################################################
#additional models that I used during training process


# run a random foret model
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

data_random_order, label_random_order = shuffle(data_grouped, data_label, random_state=42)
feature_names = data_random_order.columns

est_random_forest = RandomForestClassifier()
cv = 10
n_jobs = 2
n_estimators = [100, 300]
max_depth = [10,20]
min_samples_leaf = [2]  
max_features = ["sqrt"]  
max_leaf_nodes = [None] 

param_grid = {
    "n_estimators": n_estimators,
    "max_depth": max_depth,
    "min_samples_leaf": min_samples_leaf,
    #"max_features": max_features,
    "max_leaf_nodes": max_leaf_nodes
}

gs_rf = GridSearchCV(
    est_random_forest,
    param_grid=param_grid,
    cv=cv,
    n_jobs=n_jobs
)
gs_rf_model = gs_rf.fit(data_random_order, label_random_order.iloc[:, 1])

print("Best: %f using %s" % (gs_rf_model.best_score_, gs_rf_model.best_params_))
means = gs_rf_model.cv_results_['mean_test_score']
stds = gs_rf_model.cv_results_['std_test_score']
params = gs_rf_model.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
    
######################################################
# run a neural network model
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense

#data_random_order, label_random_order = shuffle(data_grouped, data_label, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(data_random_order, label_random_order, test_size=0.2, random_state=42)

X_train = X_train.astype('float32')
y_train = y_train['target'].astype('float32')
X_test = X_test.astype('float32')
y_test = y_test['target'].astype('float32')

model = tf.keras.Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


######################################################
# run an ensamble model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

#data_random_order, label_random_order = shuffle(data_grouped, data_label, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(data_random_order, label_random_order, test_size=0.2, random_state=42)

X_train = X_train.astype('float32')
y_train = y_train['target'].astype('float32')
X_test = X_test.astype('float32')
y_test = y_test['target'].astype('float32')

base_classifier = MLPClassifier(hidden_layer_sizes=(64, 64), activation='relu', solver='adam', max_iter=500)

ensemble_model = BaggingClassifier(base_classifier, n_estimators=10)

ensemble_model.fit(X_train, y_train)

accuracy = ensemble_model.score(X_test, y_test)
print("Test Accuracy:", accuracy)
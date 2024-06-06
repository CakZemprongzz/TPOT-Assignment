import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: 0.8093072448439977
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=KNeighborsClassifier(n_neighbors=18, p=1, weights="uniform")),
    StackingEstimator(estimator=KNeighborsClassifier(n_neighbors=18, p=1, weights="uniform")),
    MinMaxScaler(),
    StackingEstimator(estimator=MLPClassifier(alpha=0.01, learning_rate_init=0.001)),
    KNeighborsClassifier(n_neighbors=18, p=1, weights="distance")
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

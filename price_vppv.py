from __future__ import print_function

import math

from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

my_dataframe = pd.read_csv(
    "http://webhdfs.58corp.com/webhdfs/v1/home/hdp_anjuke_bi/warehouse/hdp_anjuke_dm_db.db/wzy_playground/000000_0?op=open&user.name=hdp_anjuke_bi",
    sep=",")
my_dataframe.columns = ['region_id', 'subregion_id', 'price', 'vppv']

my_dataframe = my_dataframe.head(20000)
my_dataframe = my_dataframe.reindex(np.random.permutation(my_dataframe.index))

print("\nDataframe summary:")
display.display(my_dataframe.describe())


def preprocess_examples(dataset):
    selected_features = dataset[["price"]]
    processed_features = selected_features.copy()
    return processed_features


def preprocess_targets(dataset):
    output_labels = pd.DataFrame()
    output_labels["vppv"] = (dataset["vppv"] / 10)
    return output_labels


training_examples = preprocess_examples(my_dataframe.head(10000))
training_targets = preprocess_targets(my_dataframe.head(10000))

validation_examples = preprocess_examples(my_dataframe.tail(5000))
validation_targets = preprocess_targets(my_dataframe.tail(5000))

print("\nTraining examples summary:")
display.display(training_examples.describe())
print("\nValidation examples summary:")
display.display(validation_examples.describe())

print("\nTraining targets summary:")
display.display(training_targets.describe())
print("\nValidation targets summary:")
display.display(validation_targets.describe())


def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_nn_regression_model(
        learning_rate,
        steps,
        batch_size,
        hidden_units,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    periods = 10
    steps_per_period = steps / periods

    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    dnn_regressor = tf.estimator.DNNRegressor(
        feature_columns=construct_feature_columns(training_examples),
        hidden_units=hidden_units,
        optimizer=my_optimizer
    )

    training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets["vppv"],
                                            batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                    training_targets["vppv"],
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                      validation_targets["vppv"],
                                                      num_epochs=1,
                                                      shuffle=False)

    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        dnn_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))

        print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))

        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print("Model training finished.")

    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()

    print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
    print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)

    return dnn_regressor


dnn_regressor = train_nn_regression_model(
    learning_rate=0.01,
    steps=500,
    batch_size=10,
    hidden_units=[10, 10],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

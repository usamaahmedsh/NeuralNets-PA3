"""
The main code for the feedforward networks assignment.
See README.md for details.
"""

from typing import Tuple, Dict
import tensorflow


def create_auto_mpg_deep_and_wide_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                tensorflow.keras.models.Model]:
    """Creates one deep neural network and one wide neural network.
    The networks should have the same (or very close to the same) number of
    parameters and the same activation functions.

    The neural networks will be asked to predict the number of miles per gallon
    that different cars get. They will be trained and tested on the Auto MPG
    dataset from:
    https://archive.ics.uci.edu/ml/datasets/auto+mpg

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (deep neural network, wide neural network)
    """
    # Deep Network
    deep_model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Input(shape=(n_inputs,)),
        tensorflow.keras.layers.Dense(64, activation='relu'),
        tensorflow.keras.layers.Dense(32, activation='relu'),
        tensorflow.keras.layers.Dense(32, activation='relu'),
        tensorflow.keras.layers.Dense(n_outputs)
    ])

    # Wide Network
    wide_model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Input(shape=(n_inputs,)),
        tensorflow.keras.layers.Dense(74, activation='relu'),
        tensorflow.keras.layers.Dense(38, activation='relu'),
        tensorflow.keras.layers.Dense(n_outputs)
    ])

    # Compile models
    deep_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    wide_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return deep_model, wide_model


def create_delicious_relu_vs_tanh_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                tensorflow.keras.models.Model]:
    """Creates one neural network where all hidden layers have ReLU activations,
    and one where all hidden layers have tanh activations. The networks should
    be identical other than the difference in activation functions.

    The neural networks will be asked to predict the 0 or more tags associated
    with a del.icio.us bookmark. They will be trained and tested on the
    del.icio.us dataset from:
    https://github.com/dhruvramani/Multilabel-Classification-Datasets

    which is a slightly simplified version of:

    https://archive.ics.uci.edu/ml/datasets/DeliciousMIL%3A+A+Data+Set+for+Multi-Label+Multi
    -Instance+Learning+with+Instance+Labels

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (ReLU neural network, tanh neural network)
    """
    # ReLU Network
    relu_model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Input(shape=(n_inputs,)),
        tensorflow.keras.layers.Dense(64, activation='relu'),
        tensorflow.keras.layers.Dense(32, activation='relu'),
        tensorflow.keras.layers.Dense(n_outputs, activation='sigmoid')
    ])

    # Tanh Network
    tanh_model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Input(shape=(n_inputs,)),
        tensorflow.keras.layers.Dense(64, activation='tanh'),
        tensorflow.keras.layers.Dense(32, activation='tanh'),
        tensorflow.keras.layers.Dense(n_outputs, activation='sigmoid')
    ])

    # Compile models
    relu_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    tanh_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return relu_model, tanh_model


def create_activity_dropout_and_nodropout_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                tensorflow.keras.models.Model]:
    """Creates one neural network with dropout applied after each layer, and
    one neural network without dropout. The networks should be identical other
    than the presence or absence of dropout.

    The neural networks will be asked to predict which one of six activity types
    a smartphone user was performing. They will be trained and tested on the
    UCI-HAR dataset from:
    https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (dropout neural network, no-dropout neural network)
    """
    # Dropout Network
    dropout_model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Input(shape=(n_inputs,)),
        tensorflow.keras.layers.Dense(128, activation='relu'),
        tensorflow.keras.layers.Dropout(0.4),
        tensorflow.keras.layers.Dense(64, activation='relu'),
        tensorflow.keras.layers.Dropout(0.4),
        tensorflow.keras.layers.Dense(n_outputs, activation='softmax')
    ])

    # No Dropout Network
    nodropout_model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Input(shape=(n_inputs,)),
        tensorflow.keras.layers.Dense(128, activation='relu'),
        tensorflow.keras.layers.Dense(64, activation='relu'),
        tensorflow.keras.layers.Dense(n_outputs, activation='softmax')
    ])

    # Compile models
    dropout_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    nodropout_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return dropout_model, nodropout_model


def create_income_earlystopping_and_noearlystopping_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                Dict,
                                                tensorflow.keras.models.Model,
                                                Dict]:
    """Creates one neural network that uses early stopping during training, and
    one that does not. The networks should be identical other than the presence
    or absence of early stopping.

    The neural networks will be asked to predict whether a person makes more
    than $50K per year. They will be trained and tested on the "adult" dataset
    from:
    https://archive.ics.uci.edu/ml/datasets/adult

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (
        early-stopping neural network,
        early-stopping parameters that should be passed to Model.fit,
        no-early-stopping neural network,
        no-early-stopping parameters that should be passed to Model.fit
    )
    """
    # Early Stopping Network
    earlystopping_model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Input(shape=(n_inputs,)),
        tensorflow.keras.layers.Dense(64, activation='relu'),
        tensorflow.keras.layers.Dense(32, activation='relu'),
        tensorflow.keras.layers.Dense(n_outputs, activation='sigmoid')
    ])
    earlystopping_params = {
        'epochs': 100,
        'callbacks': [tensorflow.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    }

    # No Early Stopping Network
    noearlystopping_model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Input(shape=(n_inputs,)),
        tensorflow.keras.layers.Dense(64, activation='relu'),
        tensorflow.keras.layers.Dense(32, activation='relu'),
        tensorflow.keras.layers.Dense(n_outputs, activation='sigmoid')
    ])
    noearlystopping_params = {
        'epochs': 100
    }

    # Compile models
    earlystopping_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    noearlystopping_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return earlystopping_model, earlystopping_params, noearlystopping_model, noearlystopping_params

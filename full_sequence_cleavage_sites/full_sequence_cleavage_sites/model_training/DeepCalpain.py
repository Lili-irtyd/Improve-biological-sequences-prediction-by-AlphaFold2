import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, concatenate
from pyswarm import pso
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


# Load preprocessed features
def load_features():
    aac_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/aac.npy")
    be_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/be.npy")
    cksaap_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/cksaap.npy")
    pssm_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/pssm.npy")
    labels = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/combined_labels.npy")
    return aac_features, be_features, pssm_features, cksaap_features, labels


# Build submodule for each feature
def build_submodule(input_dim, dropout_rate=0.5, hidden_neuro=500, hidden_layers=1,
                    kernel_initializer='glorot_uniform', bias_initializer='uniform', activation='relu'):
    inputs = Input(shape=(input_dim,))
    x = inputs
    for _ in range(hidden_layers):
        x = Dense(hidden_neuro, activation=activation, kernel_initializer=kernel_initializer,
                  bias_initializer=bias_initializer)(x)
        x = Dropout(dropout_rate)(x)
    return inputs, x


# Build DeepCalpain model with customized submodules
def build_model(aac_dim, be_dim, pssm_dim, cksaap_dim, dropout_rate=0.5):
    # Sub-model 1: AAC
    aac_input, aac_output = build_submodule(aac_dim, dropout_rate, hidden_neuro=500, hidden_layers=1,
                                            kernel_initializer='glorot_uniform', bias_initializer='uniform',
                                            activation='tanh')

    # Sub-model 2: PSSM
    pssm_input, pssm_output = build_submodule(pssm_dim, dropout_rate, hidden_neuro=126, hidden_layers=5,
                                              kernel_initializer='glorot_normal', bias_initializer='uniform',
                                              activation='linear')

    # Sub-model 3: CKSAAP
    cksaap_input, cksaap_output = build_submodule(cksaap_dim, dropout_rate, hidden_neuro=488, hidden_layers=1,
                                                  kernel_initializer='zero', bias_initializer='glorot_normal',
                                                  activation='selu')

    # Sub-model 4: BE
    be_input, be_output = build_submodule(be_dim, dropout_rate, hidden_neuro=500, hidden_layers=1,
                                          kernel_initializer='lecun_uniform', bias_initializer='uniform',
                                          activation='softsign')

    # Merge all sub-models
    merged = concatenate([aac_output, pssm_output, cksaap_output, be_output])
    x = Dense(20, activation='softsign', kernel_initializer='uniform', bias_initializer='he_normal')(merged)
    x = Dropout(0.433042978)(x)
    outputs = Dense(1, activation='softmax', kernel_initializer='he_normal', bias_initializer='normal')(x)

    # Final model
    model = Model(inputs=[aac_input, be_input, pssm_input, cksaap_input], outputs=outputs)
    return model


# PSO for hyperparameter optimization
def optimize_hyperparameters(aac_dim, be_dim, pssm_dim, cksaap_dim, aac_train, be_train, pssm_train, cksaap_train, y_train, aac_val, be_val, pssm_val, cksaap_val, y_val):
    def objective_function(params):
        dropout_rate, learning_rate = params
        model = build_model(aac_dim, be_dim, pssm_dim, cksaap_dim, dropout_rate)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit([aac_train, be_train, pssm_train, cksaap_train], y_train, validation_data=([aac_val, be_val, pssm_val, cksaap_val], y_val),
                            epochs=10, batch_size=32, verbose=0)
        val_loss = history.history['val_loss'][-1]
        return val_loss
    lb = [0.1, 1e-5]  # Lower bound for dropout_rate and learning_rate
    ub = [0.5, 1e-2]  # Upper bound for dropout_rate and learning_rate
    best_params, _ = pso(objective_function, lb, ub, swarmsize=10, maxiter=10)
    return best_params



# Plot AUC curve
def plot_auc(y_true, y_pred, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/Picture/DeepCalpain_AUC_4features.svg")


# Plot Training vs Validation Loss
def plot_loss(history):
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/Picture/training_vs_validation_loss.svg")


# Main function
# Main function
def main():
    # Load data
    aac_features, be_features, pssm_features, cksaap_features, labels = load_features()

    # Data splitting using train-test split
    from sklearn.model_selection import train_test_split
    features = np.concatenate([aac_features, be_features, pssm_features, cksaap_features], axis=1)
    x_train, x_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # Separate back into individual feature sets
    aac_train, be_train, pssm_train, cksaap_train = np.split(x_train, [aac_features.shape[1],
                                                                    aac_features.shape[1] + be_features.shape[1],
                                                                    aac_features.shape[1] + be_features.shape[1] + pssm_features.shape[1]], axis=1)
    aac_val, be_val, pssm_val, cksaap_val = np.split(x_val, [aac_features.shape[1],
                                                              aac_features.shape[1] + be_features.shape[1],
                                                              aac_features.shape[1] + be_features.shape[1] + pssm_features.shape[1]], axis=1)

    aac_dim, be_dim, pssm_dim, cksaap_dim = aac_features.shape[1], be_features.shape[1], pssm_features.shape[1], \
                                            cksaap_features.shape[1]

    # Optimize hyperparameters
    best_dropout_rate, best_learning_rate = optimize_hyperparameters(aac_dim, be_dim, pssm_dim, cksaap_dim, aac_train,
                                                                     be_train, pssm_train, cksaap_train, y_train, aac_val,
                                                                     be_val, pssm_val, cksaap_val, y_val)
    print(f"Best Dropout Rate: {best_dropout_rate}, Best Learning Rate: {best_learning_rate}")

    # Build the model with optimized hyperparameters
    model = build_model(aac_dim, be_dim, pssm_dim, cksaap_dim, best_dropout_rate)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        [aac_train, be_train, pssm_train, cksaap_train], y_train, validation_data=([aac_val, be_val, pssm_val, cksaap_val], y_val),
        epochs=50, batch_size=155, verbose=1, callbacks=[tf.keras.callbacks.EarlyStopping(patience=20)]
    )

    # Plot the Training vs Validation Loss
    plot_loss(history)

    # Make predictions and plot AUC
    y_pred = model.predict([aac_val, be_val, pssm_val, cksaap_val])
    plot_auc(y_val, y_pred, title="ROC Curve for DeepCalpain")

    # Save the model
    model.save("DeepCalpain_model.h5")
    print("Model saved as DeepCalpain_model.h5.")

if __name__ == "__main__":
    main()

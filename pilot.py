import pandas as pd
import src.functions as sf
from src.constants import *
from sklearn.model_selection import train_test_split, ParameterGrid

# Load entire dataset
x, y = sf.load_dataset()

# Split into training and testing subsets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# =======================================================================
# PILOT 1: Influence of architecture on convergence
# =======================================================================

print(f" ==================================================================="
      f" ARCHITECTURE INFLUENCE"
      f" ==================================================================")

# Define the hidden layers configurations
param_grid = dict(
    hidden_layers=[(16, 8),
                   (64, 32),
                   (128, 64, 32)],
    activation=["relu"],
    learning_rate=[0.001],
    batch_size=[1024],
    weight_decay=[0.0],
    epochs=[1000],
)

configurations = list(ParameterGrid(param_grid))

for i, hyperparameters in enumerate(configurations, start=1):

    # Train and evaluate the neural network
    _, model = sf.train_and_evaluate(hyperparameters, x_train, y_train, x_test, y_test)

    # Print results for each architecture
    print()
    print(f"Architecture: {hyperparameters['hidden_layers']}")
    print(f"Training time: {model.fit_time:.2f} seconds")
    print(f"Best epoch: {model.best_epoch}")
    print(f"Best validation loss: {min(model.history['loss_valid']):.6f}")
    print(f"Final validation loss: {model.history['loss_valid'][-1]:.6f}")
    print(f"Final training loss: {model.history['loss_train'][-1]:.6f}")

    # Save results
    architecture = hyperparameters["hidden_layers"]

    df_history = pd.DataFrame(model.history)
    df_history["hidden_layers"] = str(model.hidden_layers)

    sf.save_table(df_history, f"pilot/table_pilot_architecture_{architecture}.csv")

    sf.progress_bar(i, len(configurations))
    print()

sf.plot_pilot_architecture_loss_(param_grid["hidden_layers"])

# =======================================================================
# PILOT 2: Influence of learning rate on convergence
# =======================================================================

print(f" ==================================================================="
      f" LEARNING RATE INFLUENCE"
      f" ==================================================================")

# Define the hidden layers configurations
param_grid = dict(
    hidden_layers=[(64, 32)],
    activation=["relu"],
    learning_rate=[0.0001, 0.001, 0.01],
    batch_size=[1024],
    weight_decay=[0.0],
    epochs=[1000],
)

configurations = list(ParameterGrid(param_grid))

for i, hyperparameters in enumerate(configurations, start=1):

    # Train and evaluate the neural network
    _, model = sf.train_and_evaluate(hyperparameters, x_train, y_train, x_test, y_test)

    # Print results for each learning rate
    print()
    print(f"Learning rate: {hyperparameters['learning_rate']}")
    print(f"Architecture: {hyperparameters['hidden_layers']}")
    print(f"Training time: {model.fit_time:.2f} seconds")
    print(f"Best epoch: {model.best_epoch}")
    print(f"Best validation loss: {min(model.history['loss_valid']):.6f}")
    print(f"Final validation loss: {model.history['loss_valid'][-1]:.6f}")
    print(f"Final training loss: {model.history['loss_train'][-1]:.6f}")

    # Save results
    architecture = hyperparameters["hidden_layers"]
    learning_rate = hyperparameters["learning_rate"]

    df_history = pd.DataFrame(model.history)
    df_history["hidden_layers"] = str(model.hidden_layers)
    df_history["learning_rate"] = learning_rate

    sf.save_table(df_history, f"pilot/table_pilot_learning_rate_{architecture}_{learning_rate}.csv")

    sf.progress_bar(i, len(configurations))
    print()

sf.plot_pilot_lr_loss_(param_grid["hidden_layers"][0], param_grid["learning_rate"])




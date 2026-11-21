# =====================================================================
# Grid definition and division among the group
# =====================================================================

import src.functions as sf

# Define the hyperparameter search space
param_grid = {
    "hidden_layers": [(16, 8), (64,32), (128, 64, 32)],
    "activation": ["relu", "leakyrelu", "tanh", "sigmoid"],
    "learning_rate": [0.0001, 0.001, 0.01],
    "batch_size": [256, 512, 1024],
    "weight_decay": [0.0, 0.001, 0.01]
}

# Load dataset
x,y = sf.load_dataset()

# Select assigned set of configurations
n_parts = 3
part = 1
my_grid = sf.split_grid(param_grid, n_parts, part)

print(f"Running set {part}: {len(my_grid)} configurations.")
print()

# Run grid search and save results
results = sf.run_search(my_grid, x, y, checkpoint_filename=f"grid_search/table_search_set{part}.csv")

print(results)


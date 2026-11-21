# Cell Culture Growth Modeling and Neural Network Optimization
This project develops and evaluates various machine learning models to predict cell culture behavior based on experimental 
process data. A fully connected deep neural network (DNN) is implemented using PyTorch to predict three targets:
specific growth rate, specific death rate and specific production rate. A systematic grid search is used to evaluate different 
hyperparameter combinations and identify the best-performing configuration, followed by cross-validation to assess the 
selected model's performance. The project also investigates the effects of five hyperparameters (hidden layers architecture, 
activation function, learning rate, batch size and weight decay) on model performance, and compares its predictive accuracy 
with polynomial regression. By combining predictive modeling, hyperparameter optimization and model evaluation, the project 
explores the application of machine learning to bioprocess modeling and optimization.

## Highlights
* Developed a fully connected deep neural network using PyTorch to predict three cell culture response variables from
experimental data.
* Designed pilot experiments to investigate the effects of neural network architecture and learning rate on training
convergence and computational efficiency.
* Implemented a systematic hyperparameter grid search with checkpointing and workload distribution among the group to improve computational efficiency and reproducibility.
* Evaluated model performance using RMSE, NRMSE and cross-validation to compare predictive accuracy across multiple target variables.
* Compared deep neural network predictions to polynomial regression predictions to assess different modeling approaches.

## Environment

* Python: 3.14.7
* matplotlib: 3.11.0
* numpy: 2.5.2
* pandas: 3.0.5
* scikit-learn: 1.9.0
* torch: 2.13.0

## Code Design

(to be written: 500-1500 words, organized into Heading 3 subsections)

## Analysis

(to be written: 1000-3000 words, organized into Heading 3 subsections)

### Hyperparameter Search

### Cross-Validation of the Optimal Network

### Sensitivity Analysis

### Next Steps

### Comparison with Polynomial Regression

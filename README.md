# System-Identification-Using-Machine-Learning
This is a basic that project aims to identify governing equation of a simple pendulum using kinematic data.
The following steps are to be followed:
1. Collect the data for angluar displacement, angular velocity and angular acceleration from experiments.
2. Sorted input data into angluar displacement, angular velocity and angular acceleration, sin of angular displacement and square of angular velocity to create a Pandas dataframe for data processing.
3. Calculate the correlation matrix to obtain heat map and propose different possible equations to create a hypotheses space.
4. Estimate the optimal hyper-parameter for Linear Ridge Regression model to fit the data points.
5. Compare mean-squared-error for each of the fitted hypotheses to identify the governing equation based on least error.

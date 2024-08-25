# System-Identification-Using-Machine-Learning
This is a basic project that  aims to identify governing equation of a simple pendulum using kinematic data.
The following steps are to be followed:
1. Collect over 1000 data for angluar displacement, angular velocity and angular acceleration from experiments.
2. Import the data into your program.
3. Sort the imported data into angluar displacement, angular velocity, angular acceleration, sin of angular displacement and square of angular velocity to create a Pandas dataframe for further data processing.
4. Calculate the correlation matrix to obtain heat map and propose different possible equations to create a hypotheses space.
5. Estimate the optimal hyper-parameter for Linear Ridge Regression model to fit the data points for each proposed hypotheses..
6. Compare mean-squared-error for each of the fitted hypotheses to identify the governing equation based on least error.

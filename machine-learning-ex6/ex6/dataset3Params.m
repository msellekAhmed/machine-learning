function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Define the possible values that C and Sigma can take
possible_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
possible_values_count = length(possible_values);

error_temp = Inf
for i = 1:1:possible_values_count
    for j = 1:1:possible_values_count
        
        C_temp = possible_values(i);
        sigma_temp = possible_values(j);
        
        % Get the prediction
        model = svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp)); 
        prediction = svmPredict(model, Xval);
        predictionError = mean(double(prediction ~= yval));
        
        % retain the best values for c and sigma
        if (predictionError < error_temp)
            error_temp = predictionError;
            C = C_temp;
            sigma = sigma_temp;
        end
        
    end
end







% =========================================================================

end

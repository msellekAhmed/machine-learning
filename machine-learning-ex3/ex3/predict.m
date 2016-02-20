function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Implementation Note: The matrix X contains the examples in rows. 
% When you complete the code in predict.m, you will need to add the 
% column of 1?s to the matrix. The matrices Theta1 and Theta2 contain the 
% parameters for each unit in rows. Specifically, the first row of Theta1 
% corresponds to the first hidden unit in the second layer. 
% In Octave/MAT- LAB, when you compute z(2) = ?(1)a(1), be sure that you 
% index (and if necessary, transpose) X correctly so that you get a(l) 
% as a column vector.

% Full X Matrix including x0
X0 = ones(m,1); % Column vector of 1's
X = [X0 X];    % 5000 Rows, 401 columns

% Ouput of layer 2 (hidden layer) g(z) = sigmoid(z)
Z2 = sigmoid(X*Theta1'); % 5000 rows, 25 cols

% Full op_layer2 Matrix including x0 for layer 2
Z2 = [ones(m,1) Z2]; % 5000 rows, 26 cols

% Output of layer 3 (output layer)
hypothesis = sigmoid(Z2*Theta2'); % 5000 rows, 10 cols

for c=1:1:m,
    [maxValue, maxIndex] = max(hypothesis(c,:));
    p(c) = maxIndex;
end


% =========================================================================


end

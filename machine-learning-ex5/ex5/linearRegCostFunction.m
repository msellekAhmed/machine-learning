function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% Note that X already contains the bias column from ex5.m, no need to add
% over here.

%% Get the Regularized Cost function
% Hypothesis
hypothesis = X*theta; % (12 x 2) * (2 x 1) = (12 x 1), same as y
% Squared error
squared_error = (hypothesis - y).^2; % (12 x 1)
% Cost Function
J = (1/(2*m))*sum(squared_error); % (1 x 1)
% Regularization parameter
reg_term = (lambda/(2*m))*sum(theta(2:end).^2);
% Regularized cost Function
J = J + reg_term;

%% Get the regularized gradient
grad_0 = (1/m)*( X(:,1)'*(hypothesis - y) ); % for j = 0
grad_1 = (1/m)*( X(:,2:end)'*(hypothesis - y) ) +(lambda/m)*(theta(2:end)); % for j >= 0
grad = [grad_0; grad_1];
% =========================================================================

grad = grad(:);

end

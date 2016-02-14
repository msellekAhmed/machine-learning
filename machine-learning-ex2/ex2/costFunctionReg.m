function [J, grad] = costFunctionReg(THETA, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
% J = 0;
% grad = zeros(size(THETA));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%% Constants
[rows,cols] = size(X);  
% Total no.of cols = x0..xn. For regularization we want x1 to xn i.e index
% 2 to cols
%% Compute the hypothesis
hypothesis = sigmoid(X*THETA);
%% Compute the cost
J = (1/m)*(-y'*log(hypothesis) - (1-y)'*log(1-hypothesis));
% Regularized cost.
J = J + (lambda/(2*m))*sum(THETA(2:cols).^2);
%% Compute the gradient (In regularized GD, grad is different for j=0 and j = 1..n)

% Gradient for j = 0 (we use index 1 as MATLAB starts indexes at 1)
grad_0 = (1/m)*(X(:,1)'*(hypothesis-y));
% Gradient for j = 1 to n i.e. index 2 to cols
grad_1_to_n = (1/m)*(X(:,2:cols)'*(hypothesis-y)) + (lambda/m)*THETA(2:cols);

% Final gradient
grad = [grad_0;grad_1_to_n];

% =============================================================

end

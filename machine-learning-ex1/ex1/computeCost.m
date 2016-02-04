function J = computeCost(X, y, THETA)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% If we don't use matrices here, we have to use for i:1:1:m.
% The gradientDecscent.m has been done both ways.

%% Hypothesis
% Hypothesis: h = theta0x0 + theta1x1 
% Hypothesis: h = [x0 x1]*[theta0]
%                         [theta1]
%               = X*THETA
hypothesis = X*THETA; 

%% Error Term
% Error term: errorTerm = (h - y)^2
errorTerm = (hypothesis-y).^2; % square each element in place

%% Cost function:
J = (1/(2*m))*sum(errorTerm);

% =========================================================================

end

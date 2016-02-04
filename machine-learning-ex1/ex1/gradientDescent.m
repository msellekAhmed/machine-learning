function [THETA, J_history] = gradientDescent(X, y, THETA, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================

    %% Matrix way (simultaneously update THETA)
    h = X*THETA;
    THETA = THETA - alpha*(1/m)*(X'*(h-y));

    %% Without using matrices (simultaneously update THETA)
%     theta0 = THETA(1); % Since matlab indexes start at 1
%     theta1 = THETA(2);
%     errorTerm0 = 0;
%     errorTerm1 = 0;
%     for i = 1:1:m 
%         x0 = X(i,1);
%         x1 = X(i,2);
%         errorTerm0 = errorTerm0 + ( ((theta0*x0 + theta1*x1) - y(i))*x0 );
%         errorTerm1 = errorTerm1 + ( ((theta0*x0 + theta1*x1) - y(i))*x1 );
%     end
%     theta0 = theta0 - alpha*(1/m)*errorTerm0;
%     theta1 = theta1 - alpha*(1/m)*errorTerm1;
%     THETA = [theta0;theta1];
    % ============================================================

    %% COST: (use the updated THETA to calc. Cost for each iteration
    % Save the cost J in every iteration  
    J_history(iter) = computeCost(X, y, THETA);

end

end

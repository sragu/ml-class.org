function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% cost and gradient value w/o regularization
[J_base, grad_base] = costFunction(theta, X, y);

% make theta(0) to be zero, so regression is not calculated for it.
theta_partial = [ 0; theta(2:1:end,:) ];

J = J_base + ((lambda/(2*m)) * (theta_partial'*theta_partial));

grad = grad_base + (lambda/m * theta_partial');


% =============================================================

end

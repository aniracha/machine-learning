function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
%fprintf(' %f \n', size(grad));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


prediction = 0.0;
newtheta = theta';
pred = 0.0;

for k = 1:m,
		tt = X(k, :)';
		val = newtheta * tt;
		hx = sigmoid(val);
  		prediction += -y(k) * log(hx) - (1-y(k)) * log(1 - hx);
end;

J = (1/(m))*prediction;

l = size(X,2);
for p = 1:l
	for k1 = 1:m,
			tt1 = X(k1, :)';
			val1 = newtheta * tt1;
			hx1 = sigmoid(val1);
	  		pred += (hx1 - y(k1)) * X(k1, p);
	  		
	end;
	%fprintf(' %f \n', pred);
	grad(p) = pred/m;
	pred = 0.0;
end;

% =============================================================

end

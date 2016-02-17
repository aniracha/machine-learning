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



prediction = 0.0;
newtheta = theta';
pred = 0.0;

for k = 1:m,
		tt = X(k, :)';
		val = newtheta * tt;
		hx = sigmoid(val);
  		prediction += -y(k) * log(hx) - (1-y(k)) * log(1 - hx);
end;

l = size(X,2);
y1 = 0.0;

for k2 = 2:l
	y1 += theta(k2)^2;
end;

J = (1/(m))*prediction + (lambda/(2*m)) * y1;


for k1 = 1:m,
	tt1 = X(k1, :)';
	val1 = newtheta * tt1;
	hx1 = sigmoid(val1);
	pred += (hx1 - y(k1)) * X(k1, 1);	  		
end;

grad(1) = pred/m;

pred = 0.0;

for p = 2:l
	for k1 = 1:m,
			tt1 = X(k1, :)';
			val1 = newtheta * tt1;
			hx1 = sigmoid(val1);
	  		pred += (hx1 - y(k1)) * X(k1, p);
	  		
	end;
	%fprintf(' %f \n', pred);
	grad(p) = pred/m + (lambda/m)*theta(p);
	pred = 0.0;
end;



% =============================================================

end

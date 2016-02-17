function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0.0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

prediction = 0.0;
newprediction = 0.0;
newtheta = theta';
for k = 1:m,
	for i = 1:2,
  		prediction += newtheta(i) * X(k,i);
	end;
	newprediction += power(prediction - y(k), 2);
	prediction = 0.0;
end;	
J = (1/(2*m))*newprediction;




% =========================================================================

end

function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).





%g = sigmoid(g) * (1-sigmoid(g));

m = size(z,1);
n = size(z,2);
%fprintf(' %f \n', m);
for k = 1:m,
	for i = 1:n,
		g(k,i) = sigmoid(z(k,i)) * (1-sigmoid(z(k,i)));
		%g(k,i) = 1/(1+e^(-z(k,i)));
	end;
end;








% =============================================================




end

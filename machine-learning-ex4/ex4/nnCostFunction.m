function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%fprintf(' %f \n', size(y));

%fprintf(' %f size of theta 1\n', size(Theta1));

%fprintf(' %f size of theta 2\n', size(Theta2));

%fprintf(' %f size of theta 1\n', Theta1(:,1));





prediction = 0.0;
pred = 0.0;

sizeOfTheta1 = size(Theta1,1);
sizeOfTheta2 = size(Theta2,1);
X = [ones(m, 1) X];


	for k = 1:m,

		eachY = zeros(num_labels,1);

		eachY(y(k)) = 1;
		
			eachSample = X(k, :)';
			a2 = zeros(size(Theta1,1),1);
			for u = 1:sizeOfTheta1,
				val = Theta1(u, :) * eachSample;
				hx = sigmoid(val);
		  		a2(u) = hx;
		  	end;
		  	a2 = [1; a2];

		  	for w = 1:sizeOfTheta2,
				val1 = Theta2(w, :) * a2;
				hx1 = sigmoid(val1);
				prediction += -eachY(w) * log(hx1) - (1-eachY(w)) * log(1 - hx1);
		  	end;
	  		
	end;


%J = (1/(m))*prediction;


newTheta1 = Theta1(:, 2:end);
newTheta2 = Theta2(:, 2:end);

l = size(X,2);
y1 = 0.0;

for k2 = 1:size(newTheta1,1),
	for k3 = 1:size(newTheta1,2),
		y1 += newTheta1(k2,k3)^2;
	end;
end;

for k4 = 1:size(newTheta2,1),
	for k5 = 1:size(newTheta2,2),
		y1 += newTheta2(k4,k5)^2;
	end;
end;



J = (1/(m))*prediction + (lambda/(2*m)) * y1;


bigDelta2 = zeros(sizeOfTheta2,sizeOfTheta1+1);
bigDelta1 = zeros(sizeOfTheta1,size(Theta1,2));


for k = 1:m,

	
	delta3 = zeros(num_labels,1);
	delta2 = zeros(sizeOfTheta1,1);


	%printf(' %f size of sizeOfTheta1 1\n', size(Theta1,2));

	eachSample = X(k, :)';
	a1 = X(k, :)';
	a2 = zeros(size(Theta1,1),1);
	z2 = zeros(size(Theta1,1),1);

	for u = 1:sizeOfTheta1,
		z2(u) = Theta1(u, :) * eachSample;
		a2(u) = sigmoid(z2(u));
	end;

	a2 = [1; a2];
	a3 = zeros(num_labels,1);
	z3 = zeros(size(a3,1),1);


	for w = 1:sizeOfTheta2,
		z3(w) = Theta2(w, :) * a2;
		a3(w) = sigmoid(z3(w));
		delta3(w) = a3(w) - (y(k) == w);
	end;

	z2 = [1; z2];
	delta2 = [1; delta2];
	
	delta2 = (Theta2)' * delta3 .* sigmoidGradient(z2);
	

	delta2 = delta2(2:end);
	%a2 = [1;a2];
	bigDelta2 += delta3 * (a2)';



	%%%%%% starting bigDelta1

	
	%printf(' %f size of a1 1\n', size(a1));
	%printf(' %f size of delta2 1\n', size(delta2));
	bigDelta1 += delta2 * (a1)';



	%fprintf(' %f size DDDD 1\n', size(delta3 * (z2)'));



	
end;




Theta1_grad = bigDelta1/m;
Theta2_grad = bigDelta2/m;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

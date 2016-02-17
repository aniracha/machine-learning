function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m, 1) X];


n = size(X, 2);


fprintf('HELLOOOOO X rows %f\n', m);
fprintf('HELLOOOOO X cols  %f \n', n);

fprintf('HELLOOOOO theta 1 rows  %f \n', size(Theta1, 1));
fprintf('HELLOOOOO theta 1 cols %f \n', size(Theta1, 2));

fprintf('HELLOOOOO theta 2 rows  %f \n', size(Theta2, 1));
fprintf('HELLOOOOO theta 2 cols %f \n', size(Theta2, 2));





sizeOfTheta1 = size(Theta1,1);
sizeOfTheta2 = size(Theta2,1);

c = 1;
for k = 1:m,

		eachSample = X(k, :)';
		a2 = zeros(size(Theta1,1),1);
		for u = 1:sizeOfTheta1,
			val = Theta1(u, :) * eachSample;
			hx = sigmoid(val);
	  		a2(u) = hx;
	  	end;
	  	a2 = [1; a2];

	  	maxi = 0.0;
	  	label_num = 0;

	  	for w = 1:sizeOfTheta2,
			val1 = Theta2(w, :) * a2;
			hx1 = sigmoid(val1);
	  		if hx1 > maxi
	  			maxi = hx1;
	  			label_num = w;
	  		endif;
	  	end;
	  	
	  	p(k) = label_num;



end;



% =========================================================================


end

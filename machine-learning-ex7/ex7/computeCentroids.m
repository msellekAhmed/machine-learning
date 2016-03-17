function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.

% Eg. let X = [3 6; 2 7; 5 9]
%     let idx = [1 2 1] 

    for i=1:1:K
        % Find which indexes have the current centroid
        % for i = 1, we should get [1 3] i.e. 1st and 3rd since for those 
        % indeses, idx has value 1 i.e. centroid 1
        indexes = find(idx==i);
        % calculate the new centroid values columnwise
        for j=1:1:n
            % For our eg, for i=1,j=1 we get col = [3;6]
            % For our eg, for i=1,j=2 we get col = [2;9]
            col = X(:,j); 
            centroids(i,j) = (1/length(indexes))*(sum(col(indexes)));
        end
    end
% =============================================================

end


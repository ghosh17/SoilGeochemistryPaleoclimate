%       	function [Z, colmeans, colstds] = colstd(X)
% 
%           returns the column-standardized version of the data matrix
%           along with vectors containing the column means and
%           standard deviations
%           NB: this routine fails if any element of colstds is zero
%
	function [Z, colmeans, colstds] = colstd(X)
	colmeans = mean(X);     % get the column means
	colstds  = std(X);      % get the column-wise standard deviations
	for i = 1:length(colmeans)      % for each column
		Z(:,i) = (X(:,i)-colmeans(i))/colstds(i);   % subtract mean
    end                            % and divide by standard deviation
    

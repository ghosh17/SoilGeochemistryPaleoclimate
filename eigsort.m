% EIGSORT a m-function to extract and sort eigenvectors by eigenvalues in
%   decreasing order. 
%
% Created 2008:07:30 by W. Jenkins
% Modif'd 2010:03:18 DMG typed in by me b/c Bill apparently never gave the
%                    book a copy
% Modif'd 2011:03:18 DMG fix coding error (C for R)

function [U,Lambda]=eigsort(C)
[U, Lambda]=eig(C);                             % obtain eigenvectors U, eigenvalues Lambda
[lambda, ilambda]=sort(diag(Lambda),'descend'); % sort in descending order with indices
Lambda=diag(lambda);                            % reconstruct the eigenvalue matrix
U=U(:,ilambda);                                 % reorder the eigenvectors accordingly
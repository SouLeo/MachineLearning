% Function: hw1FindEigendigits()
%
% Input Arguements:
% A is an input of size x by k
% x is the total number of px in an img. (784 for this instance)
% k is the number of training imgs. (60,000 for this instance)
%
% Output:
% m is a vector of length x and the mean column vector of A
% V is a matrix of size x by k. Contains k eigenvectors of the covar
% matrix A. Eigenvectors are sorted in descending order by eigen value.

function [m, V] = hw1FindEigendigits(A)
    [x,k] = size(A);
    
    % Mean column vector of A
    m = mean(A,2);
    
    % Subtract mean column vector from A's columns
    for i = 1:x
       % subtract mean column vector from covar matrix
       A(:,i) = A(:,i) - m;
    end
    
    % Start finding eigen properities
    covar_subspace = A'*A;
    [eig_vec_sub, eig_val_sub] = eig(covar_subspace);
    % TODO: sort eigen vectors by eigen value
    V = zeros(x,k);
    for i = 1:k
       V(:,i) = norm(A*eig_vec_sub(i));
    end
end
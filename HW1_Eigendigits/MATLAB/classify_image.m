% Function: hw1FindEigendigits()
%
% Input Arguements:
% A is an input of size x by k
% x is the total number of px in an img. (784 for this instance)
% test_img is a single column test img. (784 x 1 for this instance)
%
% Output:
% label is an integer that classifies what digit the test_img
% represents

function index = classify_image(A, test_img)
    [x,k] = size(A);
    
    % Mean column vector of A
    m = uint8(mean(A,2));
    
    % Subtract mean column vector from A's columns
    % ToDo: Explanation for why??
    for i = 1:k
       % subtract mean column vector from covar matrix
       A(:,i) = A(:,i) - m;
    end
    A = double(A);
    % Find eigen properities
    covar_subspace = A'*A;
    
    [eig_vec_sub, eig_val_sub] = eig(covar_subspace);
    
    % Sort eigen val in descending order
    eig_val_sub_vec = diag(eig_val_sub);
    [~, indices] = sort(eig_val_sub_vec,'descend');

    V = zeros(k,k);
    for i = 1:k 
       V(:,i) = eig_vec_sub(:,indices(i));
    end
    %V = normc(V);

    
    index = knnsearch(V, test_img');
    
end
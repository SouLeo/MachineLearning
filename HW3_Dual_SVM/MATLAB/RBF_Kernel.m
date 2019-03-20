function K_RBF = RBF_Kernel(X, sigma)
    [~, n] = size(X);
    K_RBF = exp(-(sum(X'.^2)'*ones(1,n) -ones(n,1)*sum(X'.^2) + 2*X*X')/(2*sigma^2));
end
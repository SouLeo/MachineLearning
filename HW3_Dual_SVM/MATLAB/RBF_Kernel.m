% https://stackoverflow.com/questions/37362258/creating-a-radial-basis-function-kernel-matrix-in-matlab
function K_RBF = RBF_Kernel(x1, x2, sigma)
    K_RBF = exp(-norm(x2-x1)^2 / (2 * (sigma^2)));
end
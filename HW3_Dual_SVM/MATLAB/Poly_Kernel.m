% https://stackoverflow.com/questions/37362258/creating-a-radial-basis-function-kernel-matrix-in-matlab
function K_poly = Poly_Kernel(x1, x2, d)
    K_poly = (1 + dot(x1, x2))^d;
end
function [L, H] = compute_LH(C, alpha_j, alpha_ind, y_j, y_ind)
    if(y_ind ~= y_j)
        L = max(0, alpha_j - alpha_ind);
        H = min(C, C - alpha_ind + alpha_j);
    else
        L = max(0, alpha_ind + alpha_j - C);
        H = min(C, alpha_ind + alpha_j);
    end
end
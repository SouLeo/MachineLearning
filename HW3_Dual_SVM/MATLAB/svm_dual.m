% https://github.com/LasseRegin/SVM-w-SMO/blob/master/SVM.py
function [support_vecs] = svm_dual(train_img_col, svm_labels, max_iter)
    [~, n_samples]= size(train_img_col);
    alphas = zeros(n_samples);
    % gamma for RBF kernel
    gamma = 0.2;
    % convergence param:
    convergence_param = 1;
    % soft vs. hard margin for classifier
    C = 100;
    % params that I need more for loop
    b = 0; 
    w = 0;
    for i = 1:max_iter
        alphas_prev = alphas;
        for j = 1:n_samples
            new_ind = get_rand_index(n_samples, j);
            
            x_ind = train_img_col(:, new_ind);
            x_j = train_img_col(:, j);
            y_ind = svm_labels(new_ind);
            y_j = svm_labels(j);
            
            k_ij = RBF_Kernel(x_ind, x_ind, gamma) + ...
                RBF_Kernel(x_j, x_j, gamma) - ...
                2*RBF_Kernel(x_ind, x_j, gamma); 
            
            if k_ij == 0
                continue
            end
            
            alpha_prime_j = alphas(j);
            alpha_prime_ind = alphas(new_ind);
            [L, H] = compute_LH(C, alpha_prime_j, alpha_prime_ind, ...
                y_j, y_ind);
            
            w = training_img_col'.* (alphas * svm_labels);
            b = mean(svm_labels - w'.*training_img_col');
            
            err_i = sign(w'.*x_ind + b) - y_ind; 
            err_j = sign(w'.*x_j + b) - y_j;

           alphas(j) = alpha_prime_j + float(y_j * (err_i - err_j))/k_ij;
           alphas(j) = max(alphas(j), L);
           alphas(j) = min(alphas(j), H);
           
           alphas(ind) = alpha_prime_ind + y_ind * y_j * ...
               (alpha_prime_j - alphas(j));
        end
        if norm(alphas-alphas_prev) < convergence_param
            break
        end
    end
    b = mean(svm_labels - w'.*training_img_col');
    % line 71
    alphas_idx = alphas(alphas > 0);
    
end
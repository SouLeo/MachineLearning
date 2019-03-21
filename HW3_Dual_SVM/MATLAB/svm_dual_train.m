% https://github.com/LasseRegin/SVM-w-SMO/blob/master/SVM.py
function [support_vecs, conv, w, b] = svm_dual_train(train_img_col, svm_labels_c, max_iter)
    conv = 0;
    training_data = train_img_col';
    svm_labels = svm_labels_c';
    [n_samples, ~]= size(training_data);
    alphas = zeros(1, n_samples);
    % gamma for RBF kernel
    gamma = 0.2;
    % convergence param:
    convergence_param = 0.01;
    % soft vs. hard margin for classifier
    C = 1000;
    % params that I need more for loop
    b = 0; 
    w = 0;
    for i = 1:max_iter
        alphas_prev = alphas;
        for j = 1:n_samples
            new_ind = get_rand_index(n_samples, j);
            
            x_ind = training_data(new_ind, :);
            x_j = training_data(j, :);
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
            
            w = mtimes(training_data', (alphas .* svm_labels)');

            b = mean(svm_labels - mtimes(w',training_data'));

            err_i = sign(mtimes(w', x_ind') + b) - y_ind; 
            err_j = sign(mtimes(w', x_j') + b) - y_j;

            alphas(j) = alpha_prime_j + y_j * (err_i - err_j)/k_ij;
            alphas(j) = max(alphas(j), L);
            alphas(j) = min(alphas(j), H);
           
            alphas(new_ind) = alpha_prime_ind + y_ind * y_j * ...
                (alpha_prime_j - alphas(j));
        end
        if norm(alphas-alphas_prev) < convergence_param
            conv = 1;
            break
        end
    end
    b = mean(svm_labels - mtimes(w',training_data'));

    alphas_idx = floor(find(alphas)/n_samples) + 1;
    support_vecs = training_data(alphas_idx, :);
end
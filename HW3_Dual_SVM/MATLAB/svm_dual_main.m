[train_img_col, svm_labels] = training_data_input('digits.mat');
[test_img_col, test_svm_labels] = test_data('digits.mat');
max_iter = 1000;

% create double for loop for C and gamma training
% g_size = 20;
% C_size = 10;
% gamma_vec = logspace(-5, 5, g_size);
% C_vec = logspace(-3, 5, C_size);
% 
% acc_mat = zeros(g_size, C_size);

% for i = 1:g_size
%     for j = 1:C_size
        [support_vecs, conv, w, b] = svm_dual_train(train_img_col, ...
            svm_labels, max_iter, 1.83298071, 0.001);
        % predictions
        y_hat = sign(mtimes(w',test_img_col) + b);
        % calc acc
        acc = calc_acc(y_hat', test_svm_labels);
%         acc_mat(i, j) = acc;
%         j
%         conv
%     end
%     i
% end
disp('program finished')
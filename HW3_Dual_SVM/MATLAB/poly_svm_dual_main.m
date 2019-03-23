%[train_img_col, svm_labels] = training_data_input('digits.mat');
[test_img_col, test_svm_labels] = test_data('digits.mat');
max_iter = 1000;

% create double for loop for C and gamma training
% d_size = 10;
% C_size = 10;
% d_vec = linspace(1, 10, d_size);
% C_vec = logspace(-3, 5, C_size);
% 
% acc_mat = zeros(d_size, C_size);

% perc = linspace(0.001, 0.2, 15);
% acc_vec = zeros(1, length(perc));
% for i = 1:d_size
%     for j = 1:C_size
% for i = 1:length(perc)
        [train_img_col, svm_labels] = training_data_input('digits.mat', 0.015);
        [support_vecs, conv, w, b] = svm_poly_train(train_img_col, ...
        svm_labels, max_iter, 1, 0.001);
        % predictions
        y_hat = sign(mtimes(w',test_img_col) + b);
        % calc acc
        acc = calc_acc(y_hat', test_svm_labels);
%         acc_vec(i) = acc;
% end
%         acc_mat(i, j) = acc;
%         j
%         conv
%     end
%     i
% end
disp('program finished')
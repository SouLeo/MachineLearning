[train_img_col, svm_labels] = training_data_input('digits.mat');
[test_img_col, test_svm_labels] = test_data('digits.mat');
max_iter = 10000;
[support_vecs, conv, w, b] = svm_dual_train(train_img_col, svm_labels, max_iter); 
% predictions
y_hat = sign(mtimes(w',test_img_col) + b);
% calc acc
acc = calc_acc(y_hat', test_svm_labels);
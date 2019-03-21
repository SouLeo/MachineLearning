[train_img_col, svm_labels] = training_data_input('digits.mat');
max_iter = 1000;
[support_vecs, conv, w, b] = svm_dual_train(train_img_col, svm_labels, max_iter); 

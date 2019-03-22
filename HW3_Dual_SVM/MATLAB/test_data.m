function [test_img_col, test_svm_labels] = test_data(filename)
    % loads the .mat file of 4-D matrix containing training data of handwritten numbers
    load(filename);
    size_of_test_data = size(testImages);
    % percents are normalized
    % for total_num_px in image (784 px) to be less than training data size (60000)
    % percent training data <= 0.01306
    percent_test_data_used = 1; %0.01215;

    % reshapes the 4-d matrix of size (28, 28, 1, 60000) -> size (1, 784, 1, 60000)
    test_img_col = reshape(testImages, 1, (size_of_test_data(1))^2, 1, size_of_test_data(4));
    % creates 784 x 60000 matrix by removing dimensions = 1
    img_test_data_total = squeeze(test_img_col);
    % set max training samples
    test_cutoff = floor(percent_test_data_used*size_of_test_data(4));
    test_img_col = double(img_test_data_total(:,1:test_cutoff));
    % normalize intensity values and subtract average intensity
    test_img_col = test_img_col/255;
    test_img_col = test_img_col - repmat(mean(test_img_col,1), 784, 1);
    %
    test_labels_col = double(testLabels(1:test_cutoff))';
    test_svm_labels = test_labels_col;

%    Classification problem for 7 (pos) vs rest (neg)
    test_svm_labels_neg = find(test_svm_labels==7);
    test_svm_labels_pos = find(test_svm_labels~=7);
    test_svm_labels(test_svm_labels_neg) = -1;
    test_svm_labels(test_svm_labels_pos) = 1;

    % Classification problem for 4 (pos) vs 9 (neg)
%     svm_4_ind = find(test_svm_labels == 4);
%     svm_9_ind = find(test_svm_labels == 9);
%     imgs_4 = test_img_col(:, svm_4_ind);
%     imgs_9 = test_img_col(:, svm_9_ind);
%     imgs_total = [imgs_4 imgs_9];
%     
%     
%     svm_labels_9 = test_svm_labels(svm_9_ind);
%     svm_labels_4 = test_svm_labels(svm_4_ind);
%     svm_labels_total = [svm_labels_9; svm_labels_4];
%     svm_labels_total(svm_labels_total == 9) = -1;
%     svm_labels_total(svm_labels_total == 4) = 1;

%     Classification problem for 0 (pos) vs 8 (neg)
%     svm_0_ind = find(test_svm_labels == 0);
%     svm_8_ind = find(test_svm_labels == 8);
%     imgs_0 = test_img_col(:, svm_0_ind);
%     imgs_8 = test_img_col(:, svm_8_ind);
%     imgs_total = [imgs_0 imgs_8];
%     
%     
%     svm_labels_8 = test_svm_labels(svm_8_ind);
%     svm_labels_0 = test_svm_labels(svm_0_ind);
%     svm_labels_total = [svm_labels_8; svm_labels_0];
%     svm_labels_total(svm_labels_total == 8) = -1;
%     svm_labels_total(svm_labels_total == 0) = 1;


%     svm_0_ind = find(test_svm_labels == 0);
%     svm_8_ind = find(test_svm_labels == 8);
%     svm_3_ind = find(test_svm_labels == 3);
%     %
%     svm_1_ind = find(test_svm_labels == 1);
%     svm_7_ind = find(test_svm_labels == 7);
%     svm_9_ind = find(test_svm_labels == 9);
%     
%     imgs_0 = test_img_col(:, svm_0_ind);
%     imgs_8 = test_img_col(:, svm_8_ind);
%     imgs_3 = test_img_col(:, svm_3_ind);
%     %
%     imgs_1 = test_img_col(:, svm_1_ind);
%     imgs_7 = test_img_col(:, svm_7_ind);
%     imgs_9 = test_img_col(:, svm_9_ind);
%     imgs_total = [imgs_0 imgs_8 imgs_3 imgs_1 imgs_7 imgs_9];
%     
%     svm_labels_8 = test_svm_labels(svm_8_ind);
%     svm_labels_0 = test_svm_labels(svm_0_ind);
%     svm_labels_3 = test_svm_labels(svm_3_ind);
%     svm_labels_1 = test_svm_labels(svm_1_ind);
%     svm_labels_7 = test_svm_labels(svm_7_ind);
%     svm_labels_9 = test_svm_labels(svm_9_ind);
%     
%     svm_labels_total = [svm_labels_8; svm_labels_0; svm_labels_3; ...
%         svm_labels_1; svm_labels_7; svm_labels_9];
%     svm_labels_total(svm_labels_total == 1) = -1;
%     svm_labels_total(svm_labels_total == 7) = -1;
%     svm_labels_total(svm_labels_total == 9) = -1;
%     svm_labels_total(svm_labels_total == 8) = 1;
%     svm_labels_total(svm_labels_total == 0) = 1;
%     svm_labels_total(svm_labels_total == 3) = 1;
end
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
    test_labels_col= double(testLabels(1:test_cutoff))';
    test_svm_labels = test_labels_col;

    % Classification problem for 0 (pos) vs 1-9 (neg)
    test_svm_labels_neg = find(test_svm_labels);
    test_svm_labels_pos = find(~test_svm_labels);
    test_svm_labels(test_svm_labels_neg) = -1;
    test_svm_labels(test_svm_labels_pos) = 1;
end
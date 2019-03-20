function [train_img_col, svm_labels] = training_data_input(filename)
    % loads the .mat file of 4-D matrix containing training data of handwritten numbers
    load(filename);
    size_of_training_data = size(trainImages);
    % percents are normalized
    % for total_num_px in image (784 px) to be less than training data size (60000)
    % percent training data <= 0.01306
    percent_training_data_used = .01; %0.01215;

    % reshapes the 4-d matrix of size (28, 28, 1, 60000) -> size (1, 784, 1, 60000)
    training_img_col = reshape(trainImages, 1, (size_of_training_data(1))^2, 1, size_of_training_data(4));
    % creates 784 x 60000 matrix by removing dimensions = 1
    img_training_data_total = squeeze(training_img_col);
    % set max training samples
    training_cutoff = floor(percent_training_data_used*size_of_training_data(4));
    train_img_col = double(img_training_data_total(:,1:training_cutoff));
    % normalize intensity values and subtract average intensity
    train_img_col = train_img_col/255;
    train_img_col = train_img_col - repmat(mean(train_img_col,1), 784, 1);
    %
    train_labels_col= double(trainLabels(1:training_cutoff))';
    svm_labels = train_labels_col;

    % Classification problem for 0 (pos) vs 1-9 (neg)
    svm_labels_neg = find(svm_labels);
    svm_labels_pos = find(~svm_labels);
    svm_labels(svm_labels_neg) = -1;
    svm_labels(svm_labels_pos) = 1;
end

% Return train_img_col; svm_labels
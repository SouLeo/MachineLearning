function [imgs_total, svm_labels_total] = training_data_input(filename, perc)
    % loads the .mat file of 4-D matrix containing training data of handwritten numbers
    load(filename);
    size_of_training_data = size(trainImages);
    % percents are normalized
    % for total_num_px in image (784 px) to be less than training data size (60000)
    % percent training data <= 0.01306
    percent_training_data_used = perc; %0.01

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
    train_labels_col = double(trainLabels(1:training_cutoff))';
    svm_labels = train_labels_col;

    % Classification problem for 7 (pos) vs rest (neg)
%     svm_labels_neg = find(svm_labels==0);
%     svm_labels_pos = find(svm_labels~=0);
%     svm_labels(svm_labels_neg) = -1;
%     svm_labels(svm_labels_pos) = 1;

    % Classification problem for 4 (pos) vs 9 (neg)
    svm_0_ind = find(svm_labels ==0);
    svm_8_ind = find(svm_labels ==8);
    imgs_0 = train_img_col(:, svm_0_ind);
    imgs_8 = train_img_col(:, svm_8_ind);
    imgs_total = [imgs_0 imgs_8];
    
    
    svm_labels_8 = svm_labels(svm_8_ind);
    svm_labels_0 = svm_labels(svm_0_ind);
    svm_labels_total = [svm_labels_8; svm_labels_0];
    svm_labels_total(svm_labels_total == 8) = -1;
    svm_labels_total(svm_labels_total == 0) = 1;


%     % Classification problem for 0,8,3 (pos) vs 1,7,9 (neg)
%     svm_0_ind = find(svm_labels == 0);
%     svm_8_ind = find(svm_labels == 8);
%     svm_3_ind = find(svm_labels == 3);
%     %
%     svm_1_ind = find(svm_labels == 1);
%     svm_7_ind = find(svm_labels == 7);
%     svm_9_ind = find(svm_labels == 9);
%     
%     imgs_0 = train_img_col(:, svm_0_ind);
%     imgs_8 = train_img_col(:, svm_8_ind);
%     imgs_3 = train_img_col(:, svm_3_ind);
%     %
%     imgs_1 = train_img_col(:, svm_1_ind);
%     imgs_7 = train_img_col(:, svm_7_ind);
%     imgs_9 = train_img_col(:, svm_9_ind);
%     imgs_total = [imgs_0 imgs_8 imgs_3 imgs_1 imgs_7 imgs_9];
%     
%     svm_labels_8 = svm_labels(svm_8_ind);
%     svm_labels_0 = svm_labels(svm_0_ind);
%     svm_labels_3 = svm_labels(svm_3_ind);
%     svm_labels_1 = svm_labels(svm_1_ind);
%     svm_labels_7 = svm_labels(svm_7_ind);
%     svm_labels_9 = svm_labels(svm_9_ind);
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

% Return train_img_col; svm_labels
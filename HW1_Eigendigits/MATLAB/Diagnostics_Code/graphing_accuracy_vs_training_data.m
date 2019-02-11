% loads the .mat file of 4-D matrix containing training data of handwritten numbers 
load('digits.mat');
size_of_training_data = size(trainImages);
% percents are normalized
% TODO: shuffle test data b/c first half of data is easier than
% second half to classify... maybe shuffle test data into training data?
percent_test_data_used = 0.004;
% introduce testing images to classify
size_of_test_data = size(testImages);

accuracy_vec = [];
percent_vec = 0.0002:0.000995:0.04;
for percent = 0.0002:0.000995:0.04
    % for total_num_px in image (784 px) to be less than training data size (60000)
    % percent training data <= 0.01306
    percent_training_data_used = percent;
    
    training_img_col = reshape(trainImages, 1, (size_of_training_data(1))^2, 1, size_of_training_data(4));
    A_covariance_mat = squeeze(training_img_col); 
    training_cutoff = floor(percent_training_data_used*size_of_training_data(4));
    A_covariance_mat = A_covariance_mat(:,1:training_cutoff);
    
    [m ,V] = hw1FindEigendigits(A_covariance_mat);
    test_img_col = squeeze(reshape(testImages, 1, (size_of_test_data(1)^2), 1, size_of_test_data(4)));
    testing_cutoff = floor(percent_test_data_used*size_of_test_data(4));
    test_img_col = double(test_img_col(:,1:testing_cutoff));
    test_labels_col= double(testLabels(1:testing_cutoff))';
    
    guessed_labels = zeros(testing_cutoff, 1);
    
    for i = 1:testing_cutoff
        test_img_col(:,i) = test_img_col(:,i) - double(m);
        img_subspace = V'*test_img_col(:,i);
        
        %     % Reconstruct Test Images
        %     reconstruct = V*img_subspace;
        %     reconstruct = reshape(reconstruct, 28, 28);
        %     imshow(reconstruct)
        
        % knn search for closest classification?
        index = classify_image(A_covariance_mat,img_subspace);
        guessed_labels(i) = trainLabels(index);
    end
    
    % Determine Accuracy
    accuracy_vec = [accuracy_vec; accuracy(guessed_labels, test_labels_col)];
end

training_sample_size = 60000*percent_vec;

scatter(training_sample_size, accuracy_vec);
title('Amount of Training Data Used VS Testing Accuracy')
xlabel('Amount of Training Data Used')
ylabel('Accuracy')
% % Visualize eigenvectors 
% figure('NumberTitle','off','Name', 'Top 4 Eigenvectors from Training Set Size: 600')
% test_img_1 = 255*reshape(V(:,1),28,28);
% subplot(2,2,1)
% imshow(test_img_1)
% title('First Eigenvector')
% subplot(2,2,2)
% test_img_2 = 255*reshape(V(:,2),28,28);
% imshow(test_img_2)
% title('Second Eigenvector')
% subplot(2,2,3)
% test_img_3 = 255*reshape(V(:,3),28,28);
% imshow(test_img_3)
% title('Third Eigenvector')
% subplot(2,2,4)
% test_img_4 = 255*reshape(V(:,4),28,28);
% imshow(test_img_4)
% title('Four Eigenvector')

% % Reconstruct Test Images
% reconstruct = V*img_subspace;
% reconstruct = reshape(reconstruct, 28, 28);
% imshow(reconstruct)

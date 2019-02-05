% loads the .mat file of 4-D matrix containing training data of handwritten numbers 
load('digits.mat');
size_of_training_data = size(trainImages);
% percents are normalized
% for total_num_px in image (784 px) to be less than training data size (60000)
% percent training data <= 0.01306
percent_training_data_used = 0.002; 

% reshapes the 4-d matrix of size (28, 28, 1, 60000) -> size (1, 784, 1, 60000)
training_img_col = reshape(trainImages, 1, (size_of_training_data(1))^2, 1, size_of_training_data(4));
% creates 784 x 60000 matrix by removing dimensions = 1
A_covariance_mat = squeeze(training_img_col);
% set max training samples
training_cutoff = floor(percent_training_data_used*size_of_training_data(4));
A_covariance_mat = A_covariance_mat(:,1:training_cutoff);

% returns: 
% m, mean column vector of A
% V, matrix of eigenvectors sorted in descending order
[m ,V] = hw1FindEigendigits(A_covariance_mat);

% introduce testing images to classify
size_of_test_data = size(testImages);
% TODO: shuffle test data b/c first half of data is easier than
% second half to classify... maybe shuffle test data into training data?
percent_test_data_used = 0.002;
test_img_col = squeeze(reshape(testImages, 1, (size_of_test_data(1)^2), 1, size_of_test_data(4)));
testing_cutoff = floor(percent_test_data_used*size_of_test_data(4));
test_img_col = double(test_img_col(:,1:testing_cutoff));

guessed_labels = zeros(testing_cutoff, 1);

for i = 1:testing_cutoff
    test_img_col(:,i) = test_img_col(:,i) - double(m);
    img_subspace = V'*test_img_col(:,i);
    % knn search for closest classification?
    index = classify_image(A_covariance_mat,img_subspace);
    guessed_labels(i) = trainLabels(index); 
end
% testing eigenvectors 
% test_img = 255*reshape(V(:,1),28,28);
% imshow(test_img)

% loads the .mat file of 4-D matrix containing training data of handwritten numbers 
load('digits.mat');
size_of_training_data = size(trainImages);
% percents are normalized
% for total_num_px in image (784 px) to be less than training data size (60000)
% percent training data = 0.01306
percent_training_data_used = 1; 

% reshapes the 4-d matrix of size (28, 28, 1, 60000) -> size (1, 784, 1, 60000)
training_img_col = reshape(trainImages, 1, (size_of_training_data(1))^2, 1, size_of_training_data(4));
% creates 784 x 60000 matrix by removing dimensions = 1
A_covariance_mat = squeeze(training_img_col);
% set max training samples
cutoff = floor(percent_training_data_used*size_of_training_data(4));
A_covariance_mat(:,1:cutoff);

% returns 
% m, mean column vector of A
% V, matrix of eigenvectors sorted in descending order
[m ,V] = hw1FindEigendigits(A_covariance_mat);
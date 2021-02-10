% add noise
[train_y, train_x] = get_data('mnist_train.csv');
[test_y, test_x] = get_data('mnist_test.csv');

[N, d] = size(train_x);
R = randn(N, d); % using Gaussian noise in this experiment
train_x = [train_x, R];

[N, d] = size(test_x);
R = randn(N, d); % using Gaussian noise in this experiment
test_x = [test_x, R];

dimension = 10;
knn1_test = zeros(dimension, 1);
svm_test = zeros(dimension, 1);
dimensions = zeros(dimension, 1);

for i = 1 : dimension
    dimensions(i) = 2^i;
    [train_vector, train_mean] = task1(train_x, 2^i);
    res_train_x = (train_x - train_mean) * train_vector;
    results_testx = (test_x - train_mean) * train_vector;
    
    % 1NN
    knn = fitcknn(res_train_x, train_y, 'NumNeighbors', 1);
    predict_label2 = predict(knn, results_testx);
    knn1_test(i) = length(find(predict_label2 == test_y)) / length(test_y);
    
    % SVM
    libsvm_model = svmtrain(train_y, res_train_x, '-t 0');
    libsvm_w = libsvm_model.SVs' * libsvm_model.sv_coef;
    libsvm_b = -libsvm_model.rho;
    % 
    [testLib,test_accuracy,prob_estimates_test_lib] = svmpredict(test_y,results_testx,libsvm_model);
    svm_test(i) = test_accuracy(1);
    fprintf("iteration: %d\n", i);
end

% 1nn acc
figure
hold on
plot(dimensions, knn1_test, 'b'); % test
grid on
title('1NN Accuracy curves');
xlabel('Dimensions');
ylabel('Accuracy');
legend('Testing acc');
box on
hold off

% svm acc
figure
hold on
plot(dimensions, svm_test, 'b'); % test
grid on
title('SVM Accuracy curves');
xlabel('Dimensions');
ylabel('Accuracy');
legend('Testing acc');
box on
hold off

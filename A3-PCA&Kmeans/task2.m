% task 2: use 1nn to test
% read data
[train_y, train_x] = get_data('mnist_train.csv');
[test_y, test_x] = get_data('mnist_test.csv');

dimension = 8;
% result_train = zeros(dimension, 1);
result_test = zeros(dimension, 1);
dimension_array = zeros(dimension, 1);
for i = 1:dimension
    dimension_array(i) = 2^i;
    [train_vector, train_mean] = task1(train_x, 2^i);
    res_train_x = (train_x - train_mean) * train_vector;
    
%     [test_vector, test_mean] = PCA(test_x, 2^i);
    results_testx = (test_x - train_mean) * train_vector;
    
    knn = fitcknn(res_train_x, train_y, 'NumNeighbors', 1);
%   for train acc
%     predict_label = predict(knn, results_trainx);
%     train_accuracy = length(find(predict_label == train_y)) / length(train_y);
%     result_train(i) = train_accuracy;
%   for test acc
    predict_label2 = predict(knn, results_testx);
    result_test(i) = length(find(predict_label2 == test_y)) / length(test_y);
    
end

% acc
figure
hold on
% yPos = 'DisplayName';
% xx = result_test;
% plot(xPos, yPos, 'b'); % train
plot(dimension_array, result_test, 'b'); % test
grid on
title('PCA Accuracy Curves');
xlabel('Dimensions');
ylabel('Accuracy');
legend('Testing acc');
box on
hold off

% error
figure
hold on
% yPos = 'DisplayName';
% xx = result_test;
% plot(xPos, yPos, 'b'); % train
plot(dimension_array, 1 - result_test, 'b'); % test
grid on
title('PCA Error Curves');
xlabel('Dimensions');
ylabel('Error rate');
legend('Testing acc');
box on
hold off



t = cputime; 
% load wdbc_data
[X, Y, xTrain, yTrain, xTest, yTest] = load_data("data/wdbc_data.csv");

[number, dimension] = size(xTrain);
 
% primal训练及测试
epoch_num = 100;
% train_accs
train_accs = ones(epoch_num,1);
% test_accs
test_accs = ones(epoch_num,1);

for i = 1 : epoch_num
    svm_models = svm_train_primal(xTrain, yTrain, i);
    train_accuracy = svm_predict_primal(xTrain, yTrain, svm_models);
    test_accuracy = svm_predict_primal(xTest, yTest, svm_models);
    train_accs(i) = train_accuracy;
    test_accs(i) = test_accuracy;
end

T = cputime - t;
fprintf("svm time is %fs\n", T); 

%plot err results
figure
hold on
xPos = 1 - train_accs';
yPos = 'DisplayName';
xx = 1- test_accs';
plot(xPos ,yPos,'b'); % train
plot(xx, yPos,'r'); % test
grid on
title('SVM Curves','Color','b');
xlabel('Hyper-parameter C');
ylabel('Error Rate');
legend('Training error','Testing error');
box on
hold off


%plot acc results
figure
hold on
xPos = train_accs;
yPos = 'DisplayName';
xx = test_accs;
plot(xPos ,yPos,'b'); % train
plot(xx, yPos,'r'); % test
grid on
title('SVM Curves','Color','b');
xlabel('Hyper-parameter C');
ylabel('Accuracy');
legend('Training accuracy','Testing accuracy');
box on
hold off

t = cputime; 
% load wdbc_data
[X, Y, xTrain, yTrain, xTest, yTest] = load_data("data/wdbc_data.csv");

[number, dimension] = size(xTrain);
 

libsvm_model = svmtrain(yTrain, xTrain, '-t 0');
libsvm_w = libsvm_model.SVs' * libsvm_model.sv_coef;
libsvm_b = -libsvm_model.rho;
% % 
[trainLib,train_accuracy,prob_estimates_train_lib] = svmpredict(yTrain,xTrain,libsvm_model);

Train = cputime - t;
fprintf("libsvm training-time is %fs\n", Train); 

[testLib,test_accuracy,prob_estimates_test_lib] = svmpredict(yTest,xTest,libsvm_model);



Test = cputime - t - Train;
fprintf("libsvm testing-time is %fs\n", Test); 
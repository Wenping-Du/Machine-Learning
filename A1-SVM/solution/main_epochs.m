% read csv dataset
[data_train, label_train] = get_data("data/train.csv");
[data_test, label_test]= get_data("data/test.csv");

% primal训练及测试
epoch_num = 100;
% cs
cs = ones(epoch_num,1);
% bs
bs = ones(epoch_num,1);
% ws
ws = ones(epoch_num,1);
% train_accs
train_accs = ones(epoch_num,1);
% test_accs
test_accs = ones(epoch_num,1);
for i = 1 : epoch_num
    fprintf("iteration: %d\n", i);
%     j = 0;
%     if(i == 1)
%         j = i;
%     else 
%         j = (i - 1) * 5;
%     end
    j = i;
    svm_models = svm_train_primal(data_train, label_train, j);
    train_accuracy = svm_predict_primal(data_train, label_train, svm_models);
    test_accuracy = svm_predict_primal(data_test, label_test, svm_models);
    cs(i) = j; 
    bs(i) = svm_models.b;
    ws(i) = mean(svm_models.w);
    train_accs(i) = train_accuracy;
    test_accs(i) = test_accuracy;
end
fprintf("c: ");
fprintf("%d,", cs); 

fprintf("\nb: ");
fprintf("%f,", bs); 

fprintf("\nw: ");
fprintf("%f,", ws); 

fprintf("\ntrain: ");
fprintf("%f,", train_accs); 

fprintf("\ntest: ");
fprintf("%f,", test_accs); 

fprintf("\n");




% dual训练及测试
% svm_models = svm_train_dual(data_train, label_train, 0.001);
% train_accuracy = svm_predict_dual(data_train, label_train, svm_models);
% test_accuracy = svm_predict_dual(data_test, label_test, svm_models);
% 
% libsvm_model = svmtrain(label_train, data_train, '-t 0');
% libsvm_w = libsvm_model.SVs' * libsvm_model.sv_coef;
% libsvm_b = -libsvm_model.rho;
% 
% [trainLib,train_accuracy,prob_estimates_train_lib] = svmpredict(label_train,data_train,libsvm_model);
% [testLib,test_accuracy,prob_estimates_test_lib] = svmpredict(label_test,data_test,libsvm_model);
% 
% fprintf("b is %f\n", libsvm_b); 
% fprintf("w is %f\n", mean(libsvm_w)); 

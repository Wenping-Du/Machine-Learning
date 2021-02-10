% read csv dataset
[data_train, label_train] = get_data("data/train.csv");
[data_test, label_test]= get_data("data/test.csv");

svm_models = svm_train_primal(data_train, label_train, 1);
train_accuracy = svm_predict_primal(data_train, label_train, svm_models);
test_accuracy = svm_predict_primal(data_test, label_test, svm_models);
% %     
% % 
% % dual训练及测试
% svm_models = svm_train_dual(data_train, label_train, 1);
% train_accuracy = svm_predict_dual(data_train, label_train, svm_models);
% test_accuracy = svm_predict_dual(data_test, label_test, svm_models);
% %  
% libsvm_model = svmtrain(label_train, data_train, '-t 0');
% libsvm_w = libsvm_model.SVs' * libsvm_model.sv_coef;
% libsvm_b = -libsvm_model.rho;
% % 
% [trainLib,train_accuracy,prob_estimates_train_lib] = svmpredict(label_train,data_train,libsvm_model);
% [testLib,test_accuracy,prob_estimates_test_lib] = svmpredict(label_test,data_test,libsvm_model);
% % 
% fprintf("b is %f\n", libsvm_b); 
% fprintf("w is %f\n", mean(libsvm_w)); 

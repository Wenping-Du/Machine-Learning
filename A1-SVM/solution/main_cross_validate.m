% read csv dataset
[data_train, label_train] = get_data("data/train.csv");

% 交叉验证 for primal
C_number = 100;
% cs
cs = ones(C_number,1);
% train_accs
train_accs = ones(C_number,1);
% test_accs
test_accs = ones(C_number,1);
for c = 1 : C_number
    K = 10;
    sum_accuracy_train = 0;
    sum_accuracy_test = 0;
    for i = 1:K
        [tra_set_x, tra_set_y, val_set_x, val_set_y] = cross_data(data_train, label_train, i, K)
        svm_model = svm_train_primal(tra_set_x, tra_set_y, c);
        train_accuracy = svm_predict_primal(tra_set_x, tra_set_y, svm_model);
        test_accuracy = svm_predict_primal(val_set_x, val_set_y, svm_model);
        sum_accuracy_train = sum_accuracy_train + train_accuracy; 
        sum_accuracy_test = sum_accuracy_test + test_accuracy; 
    end

    mean_accuracy_train = sum_accuracy_train / K;
    mean_accuracy_test = sum_accuracy_test / K;
    fprintf("mean_accuracy_train: %f, mean_accuracy_test: %f\n", mean_accuracy_train, mean_accuracy_test);
    
    cs(c) = c; 
    train_accs(c) = mean_accuracy_train;
    test_accs(c) = mean_accuracy_test;
end

fprintf("c: ");
fprintf("%d,", cs); 

fprintf("\ntrain: ");
fprintf("%f,", train_accs); 

fprintf("\ntest: ");
fprintf("%f,", test_accs); 

fprintf("\n");


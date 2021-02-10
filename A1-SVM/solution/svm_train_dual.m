% # train the svm in the primal
function svm_model = svm_train_dual(data_train, label_train, regularisation_para_C)
    [number, dimension] = size(data_train);
    cvx_begin %dual
    cvx_solver sedumi;
        variables a(number);
        maximize(sum(a) - 1 / 2 * sum((a .* label_train)' * (data_train * data_train') * (a .* label_train)));
        subject to
            sum(a .* label_train)' == 0;
            0 <= a <= regularisation_para_C / number;
    cvx_end
    
    w = ((a .* label_train)' * data_train)';
    index = find((a < regularisation_para_C) & (a > 0));
    b = 1 / label_train(index)- w' * data_train(index, :)';
    svm_model.w = w;
    svm_model.b = mean(b); 
    fprintf("b is %f\n", mean(b)); 
    fprintf("w is %f\n", mean(w));
end
% # test on test data
function test_accuracy = svm_predict_dual(data_test, label_test, svm_model)
%   get the predicted Y
    Ypre = data_test * svm_model.w + svm_model.b;
%   account the accurate num
    count = 0;
    
    [num, dim] = size(label_test);
    for i = 1:num
        if (Ypre(i) * label_test(i) > 0)
            count = count + 1;
        end
    end 
    test_accuracy = 1.0 * count / num;
    fprintf("Accuracy is %f\n", test_accuracy); 
end




% load wdbc_data
[X, Y, xTrain, yTrain, xTest, yTest] = load_data("data/wdbc_data.csv");

iteration_num = 100;
% adaboost training
t = cputime; 
output = adaboost(xTrain, yTrain, iteration_num);
Train = cputime - t;
fprintf("adaboost training-time is %fs\n", Train); 

% testing
sum_Hx = zeros(size(yTest));
testError = zeros(iteration_num,1);
testAcc = zeros(iteration_num,1);
for i = 1 : iteration_num
    hx = double(output(i).direct * xTest(:, output(i).dimen) < output(i).s);
    hx(hx == 0) = -1;
    sum_Hx = sum_Hx + hx * output(i).alpha;
    Hx = sign(sum_Hx);
    testError(i) = sum(Hx ~= yTest) / length(xTest);
    testAcc(i) = 1 - testError(i);
end
Test=cputime - t - Train;
fprintf("adaboost testing-time is %fs\n", Test); 


%plot err results
figure
hold on
xPos = [output.errorRate]';
yPos = 'DisplayName';
xx = testError;
plot(xPos, yPos,'b'); % train
plot(xx, yPos,'r'); % test
grid on
title('AdaBoost Curves','Color','b');
xlabel('Rounds');
ylabel('Error Rate');
legend('Training error','Testing error');
box on
hold off


%plot acc results
figure
hold on
xPos = 1 - [output.errorRate]';
yPos = 'DisplayName';
xx = testAcc;
plot(xPos ,yPos,'b'); % train
plot(xx, yPos,'r'); % test
grid on
title('AdaBoost Curves','Color','b');
xlabel('Rounds');
ylabel('Accuracy');
legend('Training accuracy','Testing accuracy');
box on
hold off


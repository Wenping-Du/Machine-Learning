iteration_num = 100;
t = cputime; 
% load wdbc_data
[X, Y, xTrain, yTrain, xTest, yTest] = load_data("data/wdbc_data.csv");
%fitensemble
% plot err-rate results
figure
adaTrain = fitensemble(xTrain,yTrain,'AdaBoostM1',iteration_num,'tree');
trainLoss = resubLoss(adaTrain,'mode','cumulative');

Train = cputime - t;
fprintf("fitensemble training-time is %fs\n", Train); 

plot(trainLoss);
hold on
adaTest = fitensemble(X, Y, 'AdaBoostM1', iteration_num, 'tree', 'type', 'classification', 'Holdout', 0.5);
testkloss = kfoldLoss(adaTest,'mode','cumulative');

Test=cputime - t - Train;
fprintf("fitensemble testing-time is %fs\n", Test); 

plot(testkloss);
grid on
title('fitensemble Curves','Color','b')
xlabel('Rounds')
ylabel('Error Rate')
legend('Training error','Testing error');
box on
hold off;


%plot acc results
figure
hold on
xPos = 1 - trainLoss;
yPos = 'DisplayName';
xx = 1 - testkloss;
plot(xPos ,yPos,'b'); % train
plot(xx, yPos,'r'); % test
grid on
title('fitensemble Curves','Color','b');
xlabel('Rounds');
ylabel('Accuracy');
legend('Training accuracy','Testing accuracy');
box on
hold off

%load training dataset, and extract X and y.

data = importdata('wdbc_data.csv');
[num1,dim1]=size(data.data);
X=data.data;

[~, ~, Y] = unique(data.textdata(:,2));
Y(Y == 2) = -1;

%   get X 
train_x = X(1:300, :);
test_x = X(301:end, :);

%   get Y
train_y  = Y(1:300, :);
test_y = Y(301:end, :);

%initialize the distribution Dt as uniform distribution
Dt=ones(300,1) / 300;
%require the user to input the number of iterations
T = 50;
alphat=zeros(T,1);%store weights for weak hypothesis ht
threshold=zeros(T,1);%store thresholds of ht
direction=zeros(T,1);%store directions of ht,direction of -1 -> 1 change
ind=zeros(T,1);%store which feature is chosen for ht

H=zeros(300,1); %weighted sum of weak hypotheses
train_acc = zeros(T,1);


test_acc = zeros(T,1);
D_test=ones(269,1) / 269;
H_test=zeros(269,1); 

for t=1:T
    %stump represents weak hypothesis ht
    [stump] = build_stump(train_x,train_y,Dt);
    errort=stump.werr;% weighted error
    alphat(t)=0.5*log((1-errort)/errort);
    threshold(t)=stump.x0;
    direction(t)=stump.s;
    ind(t)=stump.ind;
    
    %update Dt
    ht_value=sign(direction(t) * (train_x(:, ind(t)) - threshold(t)));
    %ignore normalization since it's done in "build_stump"
    Dt=Dt.*exp(-alphat(t).*train_y.*ht_value);
    %sum up ht_values with weights
    H=H+alphat(t)*ht_value;  
    Hx = sign(H);
    train_acc(t) = sum(Hx == train_y) / length(train_x);
    
    
    ht_value=sign(direction(t) * (test_x(:, ind(t)) - threshold(t)));
    D_test=D_test.*exp(-alphat(t).*test_y.*ht_value);
    H_test = H_test + ht_value * alphat(t);
    Hx = sign(H_test);
    test_acc(t) = sum(Hx == test_y) / length(test_x);
end


%plot err results
figure
hold on
xPos = train_acc;
yPos = 'DisplayName';
xx = test_acc;
plot(xPos ,yPos,'b'); % train
plot(xx, yPos,'r'); % test
grid on
title('AdaBoost Curves','Color','b');
xlabel('Rounds');
ylabel('Accuracy');
legend('Training acc','Testing acc');
box on
hold off

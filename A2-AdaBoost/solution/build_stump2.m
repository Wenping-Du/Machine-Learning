function [decision_stump] = build_stump2(X,y,w,mask)
w = w./sum(w);
st = struct;
[~, dim] = size(X);
% minerror in every demension
weight_error = zeros(dim, 1);
for i = 1 : dim
    %sort
    [sortx,number] = sort(X(:,i),'ascend');
    e = y(number)' ~= mask;
    errs = double(e);
    werrs = errs * w(number);
    %Screen out the duplicate numbers to find the total split point
    sp = find(sortx(1:end-1)<sortx(2:end));  
    if(~isempty(sp))
        %two cases
        [minWeighterror1,minNumber1] = min(werrs(sp));
        [minWeighterror2,minNumber2] = min(1-werrs(sp));
        if minWeighterror1 < minWeighterror2
            st(i).weightError = minWeighterror1;
            st(i).direction = true;
            %The position of segmentation and the mean 
            if(i==1)
                st(i).threshold = sortx(1)-0.5;        
            elseif(i==dim+1)            
                st(i).threshold = sortx(dim)+0.5;        
            else
                st(i).threshold = (sortx(sp(minNumber1))+sortx(sp(minNumber1)+1))/2;      
            end

        else
            st(i).weightError = minWeighterror2;
            st(i).direction = false;
            if(i==1)
                st(i).threshold = sortx(1)-0.5;        
            elseif(i==dim+1)            
                st(i).threshold = sortx(dim)+0.5;        
            else
                st(i).threshold = (sortx(sp(minNumber2))+sortx(sp(minNumber2)+1))/2;      
            end

        end
    end
    st(i).dimension = i;
    weight_error(i) = st(i).weightError;
end
[~, number] = min(weight_error);
decision_stump = st(number);
% predict
if(decision_stump.direction == 1)
        m = X(:,decision_stump.dimension) < decision_stump.threshold;
        decision_stump.hx = double(m);
        decision_stump.hx(decision_stump.hx == 0) = -1;
elseif(decision_stump.direction ~= 1)
        n = X(:,decision_stump.dimension) >= decision_stump.threshold;
        decision_stump.hx = double(n);
        decision_stump.hx(decision_stump.hx == 0) = -1;
end
            


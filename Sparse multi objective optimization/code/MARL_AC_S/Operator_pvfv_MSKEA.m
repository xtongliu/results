function [OffDec,OffMask] = Operator_pvfv_MSKEA(Problem,ParentDec,ParentMask,pv,fv,delta,VaryGroup)
    [N,D] = size(ParentMask);
    Parent1Mask = ParentMask(1:N/2,:);
    Parent2Mask = ParentMask(N/2+1:end,:);
    OffMask = Parent1Mask;
    for i = 1 : N/2
        group = VaryGroup(i);
        if group == 1
            pv_i = pv;
        else
            pv_i = 0.8 * pv;
        end

        if rand < 0.5
            index = find(Parent1Mask(i,:)&~Parent2Mask(i,:));
            index = index(TS(-pv_i(index)));
            OffMask(i,index) = 0;
        else
            index = find(~Parent1Mask(i,:)&Parent2Mask(i,:));
            index = index(TS(pv_i(index)));
            OffMask(i,index) = Parent2Mask(i,index);
        end
    end
    
    for i = 1 : N/2
        group = VaryGroup(i);
        if group == 1
            pv_i = pv;
            fv_i = fv;
            delta_i = delta;
        else
            pv_i = 0.8 * pv;
            fv_i = 0.8 * fv;
            delta_i = 0.8 * delta;
        end

        if rand < (1-delta_i)
            f_vector = fv_i;
            f_vector(fv_i>0)=1;
            index=find(OffMask(i,:)~=f_vector);
            if rand < 0.5
                index=index(TS(-fv_i(index)));
                OffMask(i,index)=1;
            else
                index=index(TS(fv_i(index)));
                OffMask(i,index)=0;
            end
        else
            if rand < 0.5
                index = find(OffMask(i,:));
                index = index(TS(-pv_i(index)));
                OffMask(i,index) = 0;
            else
                index = find(~OffMask(i,:));
                index = index(TS(pv_i(index)));
                OffMask(i,index) = 1;
            end
        end
    end
    
    if any(Problem.encoding~=4)
        OffDec = OperatorGAhalf(Problem,ParentDec);
        OffDec(:,Problem.encoding==4) = 1;
    else
        OffDec = ones(N/2,D);
    end
end

function index = TS(pv)
% Binary tournament selection
    if isempty(pv)
        index = [];
    else
        index = TournamentSelection(2,1,pv);
    end
end

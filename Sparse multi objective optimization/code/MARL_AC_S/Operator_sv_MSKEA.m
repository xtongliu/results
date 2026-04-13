function [OffDec,OffMask] = Operator_sv_MSKEA(Problem,ParentDec,ParentMask,sv,VaryGroup)
    [N,D] = size(ParentMask);
    Parent1Mask = ParentMask(1:N/2,:);
    Parent2Mask = ParentMask(N/2+1:end,:);
    OffMask = Parent1Mask;
    for i = 1 : N/2
        group = VaryGroup(i);
        if group == 1
            sv_i = sv;
        else
            sv_i = 0.8 * sv;
        end
        rate0   = sv_i;
        rate1   = 1-rate0;
        diff = find(Parent1Mask(i,:)~=Parent2Mask(i,:));
        temp_rate1=rate1(diff);
        temp_rate0=rate0(diff);
        rate = zeros(1,length(diff));
        rate(logical(OffMask(i,diff)))  = temp_rate1(logical(OffMask(i,diff)));
        rate(logical(~OffMask(i,diff))) = temp_rate0(logical(~OffMask(i,diff)));
        exchange  = rand(1,length(diff)) < rate;
        OffMask(i,diff(exchange))=~OffMask(i,diff(exchange));
    end

    Mutation_p=1/D;
    % Mu_exchange=rand(1,D)<Mutation_p;
    Mu_exchange=rand(N/2,D)<Mutation_p;

    for i = 1 : N/2
        group = VaryGroup(i);
        if group == 1
            sv_i = sv;
        else
            sv_i = 0.8 * sv;
        end
        rate0   = sv_i;
        rate1   = 1-rate0;
        if sum(Mu_exchange(i,:))
            subscript = find(Mu_exchange(i,:)==1);
            rate = zeros(1, size(subscript,2));
            rate(logical(OffMask(i,subscript))) = rate1(subscript(logical(OffMask(i,subscript))));
            rate(logical(~OffMask(i,subscript))) = rate0(subscript(logical(~OffMask(i,subscript))));
            exchange = rand(1, size(subscript,2)) < rate;
            OffMask(i,subscript(exchange)) = ~OffMask(i,subscript(exchange));
        end
    end
   
    if any(Problem.encoding~=4)
        OffDec = OperatorGAhalf(Problem,ParentDec);
        OffDec(:,Problem.encoding==4) = 1;
    else
        OffDec = ones(N/2,D);
    end
end

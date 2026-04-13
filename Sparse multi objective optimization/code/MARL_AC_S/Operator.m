function [OffDec,OffMask] = Operator(Problem,ParentDec,ParentMask,Fitness,Mask,operatorType,Memory,sv,pv,fv,delta,paramCtrl)
[N,~]       = size(ParentDec);
Parent1Dec  = ParentDec(1:floor(end/2),:);
Parent2Dec  = ParentDec(floor(end/2)+1:floor(end/2)*2,:);
Parent1Mask = ParentMask(1:floor(end/2),:);
Parent2Mask = ParentMask(floor(end/2)+1:floor(end/2)*2,:);

VaryGroup = kmeans(Fitness',2)';
MaxGroup  = max(VaryGroup);

%% Crossover and mutation for dec
if any(Problem.encoding~=4)
    switch operatorType
        case 1 % GLP_GA (original)
            [OffDec,~,~] = GLP_OperatorGAhalf(Problem,Parent1Dec,Parent2Dec,4,paramCtrl);

            %% Crossover for mask
            OffMask = Parent1Mask;
            for i = 1 : N/2
                SelectedGroup = randi(MaxGroup,1);
                index = xor(Parent1Mask(i,:),Parent2Mask(i,:));
                if rand < 0.5
                    index = (SelectedGroup == VaryGroup) & index & rand(1,Problem.D) < 1;
                    OffMask(i,index) = 0;
                else
                    index = (SelectedGroup == VaryGroup) & index & rand(1,Problem.D) < 1;
                    OffMask(i,index) = 1;
                end
            end

            for i = 1 : N/2
                recent_fitness = Memory(max(end-10,1):end);
                fitness_diversity = length(unique(recent_fitness));
                mutation_prob = (100*mean(Mask,'all'))/(Problem.D * (fitness_diversity)^2 + eps);

                if rand < 0.5
                    index = rand(1,Problem.D) < mutation_prob;
                    OffMask(i,index) = 0;
                else
                    index = rand(1,Problem.D) < mutation_prob;
                    OffMask(i,index) = 1;
                end
            end
        case 2
            [OffDec,OffMask] = Operator_pvfv_MSKEA(Problem,ParentDec,ParentMask,pv,fv,delta,VaryGroup);
            return;
        case 3 
            [OffDec,OffMask] = Operator_sv_MSKEA(Problem,ParentDec,ParentMask,sv,VaryGroup);
            return;
    end
    OffDec(:,Problem.encoding==4) = 1;
else
    OffDec = ones(size(Parent1Dec));
end
end

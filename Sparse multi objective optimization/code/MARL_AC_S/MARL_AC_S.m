classdef MARL_AC_S < ALGORITHM
    % <multi> <real/integer/binary> <large/none> <constrained/none> <sparse>
    % Automated guiding vector selection-based evolutionary algorithm

    %------------------------------- Copyright --------------------------------
    % Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
    % research purposes. All publications which use this platform or any code
    % in the platform should acknowledge the use of "PlatEMO" and reference "Ye
    % Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
    % for evolutionary multi-objective optimization [educational forum], IEEE
    % Computational Intelligence Magazine, 2017, 12(4): 73-87".
    %--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)

            %% Population initialization
            % Calculate the fitness of each decision variable
            [Fitness1,TDec,TMask,TempPop]      = FitnessCal(Problem,5);
            [Population,Dec,Mask,FitnessSpea2] = EnvironmentalSelection(TempPop,TDec,TMask,Problem.N);



            % Agent 1: Fitness selection (3 discrete actions)

            % Agent 3: Operator selection (3 discrete actions)

            agent = MAAGENT(4, [13, 13, 13, 13], [3, 1, 3, 1], [1, 0, 1, 0], {[], [0.5, 1.0], [], [0, 1]});


            Memory = [];

            sv = 0.5*ones(1,Problem.D);
            pv = 0.5*ones(1,Problem.D);
            % fv = 0.5*ones(1,Problem.D);
            paramState = [1,20,20]; % [proC, disC, disM]

            Last_temp_num = 0;

            inFE = 0;
            [state, ~, ~] = CalStateReward(Problem, Population, Mask, Population, Mask, inFE);

            clear TempPop TDec TMask;

            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2, 2*Problem.N, FitnessSpea2);
                LastPopulation = Population;
                LastMask = Mask;


                action = agent.Action(state);


                [fitnessaction, selectratio, operatoraction, paramaction] = agent.SelectActions(action, Problem);


                paramaction = min(max(paramaction,0),1);
                focus = min(3,floor(paramaction*3)+1);
                localScale = paramaction*3 - (focus-1);
                x = 2*localScale - 1;
                eta = 0.2;
                switch focus
                    case 1
                        target = min(max(1 + 0.2*x,0.6),1.0);
                    case 2
                        target = min(max(20*exp(0.8*x),2),40);
                    otherwise
                        target = min(max(20*exp(1.0*x),5),60);
                end
                paramState(focus) = (1-eta)*paramState(focus) + eta*target;


                switch fitnessaction
                    case 1
                        Fitness = Fitness1;
                    case 2
                        Fitness = sum(Mask==0);
                        if size(Mask,1) == 1
                            Fitness = Mask;
                        end
                    case 3
                        Fitness = rand(1,Problem.D);
                end



                delta = Problem.FE / Problem.maxFE;

                if delta < 0.618
                    fv = std(Dec(FitnessSpea2==1,:),0,1);
                    fv(:,Problem.encoding==4) = sum(Mask(FitnessSpea2==1,Problem.encoding==4),1);
                end

                First_Mask = Mask(FitnessSpea2==1,:);
                [temp_num,~] = size(First_Mask);
                temp_vote = sum(First_Mask,1);
                if temp_num > 0
                    sv = (Last_temp_num/(Last_temp_num+temp_num))*sv + (temp_num/(Last_temp_num+temp_num))*(temp_vote/temp_num);
                    Last_temp_num = temp_num;
                end

                if delta < 0.618
                    pv = pv.*(1-sv)*sqrt(delta) + pv;
                end
                % ------------------------------------------


                [OffDec,OffMask] = Operator(Problem,Dec(MatingPool,:),Mask(MatingPool,:),Fitness,Mask,operatoraction,Memory,sv,pv,fv,delta,paramState);


                FinalPopSize = ceil(Problem.N + (Problem.N * selectratio));
                FinalPopSize = max(FinalPopSize, Problem.N);
                FinalPopSize = min(FinalPopSize, 2*Problem.N);


                LastFE = Problem.FE;
                Offspring = Problem.Evaluation(OffDec.*OffMask);


                [Population,Dec,Mask,FitnessSpea2] = EnvironmentalSelection([Population,Offspring],[Dec;OffDec],[Mask;OffMask],FinalPopSize);


                [~, next_state, reward] = CalStateReward(Problem, LastPopulation, LastMask, Population, Mask, LastFE);


                current_memory = fitnessaction;
                Memory = [Memory; current_memory];
                if size(Memory, 1) > 50
                    Memory = Memory(end-49:end, :);
                end


                agent.Experience(state, action, reward, next_state);


                agent.Train();


                state = next_state;
            end
        end
    end
end

classdef RLDMOEA < ALGORITHM
% <multi> <real> <dynamic>
% Reinforcement learning based dynamic multi-objective evolutionary algorithm
    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [CR,F,alpha,gamma,detectNum,tau,keepRatio,predRatio,sigma] = ...
                Algorithm.ParameterSet(0.9,0.5,0.9,0.6,10,0.2,0.2,0.6,0.05);

            %% Initialization
            Population = Problem.Initialization();
            [Population,FrontNo,CrowdDis] = EnvironmentalSelection_RLDMOEA(Population,Problem.N);
            Archive = GetNonDominated_RLDMOEA(Population);

            % RL tables: states x actions
            % s1: slight, s2: medium, s3: high
            % a1: KBP,    a2: ILS,    a3: CBP
            Q    = zeros(3,3);
            Rbar = zeros(3,3);

            % Dynamic display handling, following SGEA style
            Algorithm.save = sign(Algorithm.save)*inf;
            AllPop = [];

            %% Optimization
            while Algorithm.NotTerminated(Population)

                % -------- Detect environmental change --------
                [changed,severity,RePop] = DetectChange_RLDMOEA(Problem,Population,detectNum);

                if changed
                    % Save the old population before change for dynamic display
                    AllPop = [AllPop,Population];

                    % Historical non-dominated set under previous environment
                    OldArchive = Archive;

                    % Re-evaluate current population in the new environment
                    Population = RePop;
                    [Population,FrontNo,CrowdDis] = EnvironmentalSelection_RLDMOEA(Population,Problem.N);
                    NewArchive = GetNonDominated_RLDMOEA(Population);

                    % -------- RL decision --------
                    state  = MapState_RLDMOEA(severity);
                    action = SelectAction_RLDMOEA(Q(state,:),tau);

                    % -------- Change response --------
                    switch action
                        case 1
                            % a1: KBP
                            Population = KneeBasedPrediction_RLDMOEA( ...
                                Problem,Population,OldArchive,NewArchive, ...
                                Problem.N,keepRatio,predRatio,sigma);

                        case 2
                            % a2: ILS
                            Population = IndicatorLocalSearch_RLDMOEA( ...
                                Problem,Population,NewArchive, ...
                                Problem.N,keepRatio,predRatio,sigma);

                        case 3
                            % a3: CBP
                            Population = CenterBasedPrediction_RLDMOEA( ...
                                Problem,Population,OldArchive,NewArchive, ...
                                Problem.N,keepRatio,predRatio,sigma);
                    end

                    % Rebuild archive after response
                    [Population,FrontNo,CrowdDis] = EnvironmentalSelection_RLDMOEA(Population,Problem.N);
                    Archive = GetNonDominated_RLDMOEA(Population);

                    % Reward by HV of current ND set
                    reward    = HypervolumeReward_RLDMOEA(Archive.objs);
                    nextState = state;
                    Rbar(state,action) = 0.9*Rbar(state,action) + 0.1*reward;
                    Q(state,action)    = Q(state,action) + ...
                        alpha*(reward + gamma*max(Q(nextState,:)) - Q(state,action));
                end

                % -------- One normal generation with NSGA-II-DE --------
                [Population,FrontNo,CrowdDis] = OneGenerationNSGAIIDE_RLDMOEA( ...
                    Problem,Population,CR,F);
                Archive = GetNonDominated_RLDMOEA(Population);

                % Return all dynamic populations for visualization
                if Problem.FE >= Problem.maxFE
                    Population = [AllPop,Population];
                    [~,rank]   = sort(Population.adds(zeros(length(Population),1)));
                    Population = Population(rank);
                end
            end
        end
    end
end

%% =========================================================
function [changed,severity,RePop] = DetectChange_RLDMOEA(Problem,Population,detectNum)
    changed  = false;
    severity = 0;
    RePop    = Population;

    N = numel(Population);
    if N == 0
        return;
    end

    k = min(detectNum,N);
    idx    = randperm(N,k);
    DetOld = Population(idx);

    oldObj = DetOld.objs;
    oldCon = GetConsMatrix_RLDMOEA(DetOld);

    DetNew = Problem.Evaluation(DetOld.decs);
    newObj = DetNew.objs;
    newCon = GetConsMatrix_RLDMOEA(DetNew);

    changed = ~isequal(oldObj,newObj) || ~isequal(oldCon,newCon);

    if changed
        % Re-evaluate the whole population only after confirming the change
        RePop = Problem.Evaluation(Population.decs);

        % Severity estimation by normalized detector deviation
        utopia = min([oldObj;newObj],[],1);
        nadir  = max([oldObj;newObj],[],1);
        denom  = nadir - utopia + 1e-12;

        rel = abs(newObj-oldObj) ./ repmat(denom,k,1);
        severity = mean(sum(rel,2));
    end
end

function state = MapState_RLDMOEA(severity)
    if severity <= 1e-3
        state = 1;
    elseif severity <= 3e-3
        state = 2;
    else
        state = 3;
    end
end

function action = SelectAction_RLDMOEA(Qrow,tau)
    tau  = max(tau,1e-8);
    prob = exp((Qrow-max(Qrow))/tau);
    prob = prob./sum(prob);

    r = rand;
    c = cumsum(prob);
    action = find(r<=c,1,'first');
    if isempty(action)
        action = randi(numel(Qrow));
    end
end

%% =========================================================
function Population = KneeBasedPrediction_RLDMOEA(Problem,Population,OldArchive,NewArchive,N,keepRatio,predRatio,sigma)
    lower = reshape(Problem.lower,1,[]);
    upper = reshape(Problem.upper,1,[]);
    D     = numel(lower);
    span  = upper - lower;

    if isempty(NewArchive)
        Population = Problem.Evaluation(RandomDecs_RLDMOEA(N,lower,upper));
        return;
    end

    % Keep some current individuals
    nKeep = max(1,round(keepRatio*N));
    keepDec = SampleRows_RLDMOEA(Population.decs,nKeep,false);

    % Predict by knee movement
    nPred = max(0,round(predRatio*N));
    nPred = min(nPred,N-nKeep);
    nRand = N - nKeep - nPred;

    if isempty(OldArchive)
        predDec = SampleRows_RLDMOEA(NewArchive.decs,nPred,true);
    else
        [kOldDec,~] = KneeSolution_RLDMOEA(OldArchive);
        [kNewDec,~] = KneeSolution_RLDMOEA(NewArchive);
        shift  = kNewDec - kOldDec;
        base   = SampleRows_RLDMOEA(NewArchive.decs,nPred,true);
        noise  = randn(nPred,D).*repmat(sigma*span,nPred,1);
        predDec = base + repmat(shift,nPred,1) + noise;
        predDec = min(max(predDec,repmat(lower,nPred,1)),repmat(upper,nPred,1));
    end

    randDec = RandomDecs_RLDMOEA(nRand,lower,upper);
    newDec  = [keepDec;predDec;randDec];
    Population = Problem.Evaluation(newDec);
end

function Population = CenterBasedPrediction_RLDMOEA(Problem,Population,OldArchive,NewArchive,N,keepRatio,predRatio,sigma)
    lower = reshape(Problem.lower,1,[]);
    upper = reshape(Problem.upper,1,[]);
    D     = numel(lower);
    span  = upper - lower;

    if isempty(NewArchive)
        Population = Problem.Evaluation(RandomDecs_RLDMOEA(N,lower,upper));
        return;
    end

    nKeep = max(1,round(keepRatio*N));
    keepDec = SampleRows_RLDMOEA(Population.decs,nKeep,false);

    nPred = max(0,round(predRatio*N));
    nPred = min(nPred,N-nKeep);
    nRand = N - nKeep - nPred;

    if isempty(OldArchive)
        predDec = SampleRows_RLDMOEA(NewArchive.decs,nPred,true);
    else
        cOld   = mean(OldArchive.decs,1);
        cNew   = mean(NewArchive.decs,1);
        shift  = cNew - cOld;
        base   = SampleRows_RLDMOEA(NewArchive.decs,nPred,true);
        noise  = randn(nPred,D).*repmat(sigma*span,nPred,1);
        predDec = base + repmat(shift,nPred,1) + noise;
        predDec = min(max(predDec,repmat(lower,nPred,1)),repmat(upper,nPred,1));
    end

    randDec = RandomDecs_RLDMOEA(nRand,lower,upper);
    newDec  = [keepDec;predDec;randDec];
    Population = Problem.Evaluation(newDec);
end

function Population = IndicatorLocalSearch_RLDMOEA(Problem,Population,Archive,N,keepRatio,predRatio,sigma)
    lower = reshape(Problem.lower,1,[]);
    upper = reshape(Problem.upper,1,[]);
    D     = numel(lower);
    span  = upper - lower;

    if isempty(Archive)
        Population = Problem.Evaluation(RandomDecs_RLDMOEA(N,lower,upper));
        return;
    end

    nKeep = max(1,round(keepRatio*N));
    keepDec = SampleRows_RLDMOEA(Population.decs,nKeep,false);

    nLocal = max(0,round(predRatio*N));
    nLocal = min(nLocal,N-nKeep);
    nRand  = N - nKeep - nLocal;

    Aobj = Archive.objs;
    zmin = min(Aobj,[],1);
    zmax = max(Aobj,[],1);
    denom = zmax - zmin + 1e-12;

    localDec = zeros(nLocal,D);
    for i = 1 : nLocal
        baseID  = randi(numel(Archive));
        bestDec = Archive(baseID).decs;
        bestObj = Archive(baseID).objs;
        bestSca = sum((bestObj-zmin)./denom);

        for t = 1 : 3
            candDec = bestDec + randn(1,D).*(0.25*sigma*span);
            candDec = min(max(candDec,lower),upper);
            candSol = Problem.Evaluation(candDec);

            candSca = sum((candSol.objs-zmin)./denom);
            if Dominates_RLDMOEA(candSol.objs,bestObj) || candSca < bestSca
                bestDec = candSol.decs;
                bestObj = candSol.objs;
                bestSca = candSca;
            end
        end
        localDec(i,:) = bestDec;
    end

    randDec = RandomDecs_RLDMOEA(nRand,lower,upper);
    newDec  = [keepDec;localDec;randDec];
    Population = Problem.Evaluation(newDec);
end

%% =========================================================
function [Population,FrontNo,CrowdDis] = OneGenerationNSGAIIDE_RLDMOEA(Problem,Population,CR,F)
    N = min(numel(Population),Problem.N);
    [Population,FrontNo,CrowdDis] = EnvironmentalSelection_RLDMOEA(Population,N);

    MatingPool = TournamentSelection(2,2*N,FrontNo,-CrowdDis);
    Parent2    = Population(MatingPool(1:N));
    Parent3    = Population(MatingPool(N+1:end));

    Offspring  = OperatorDE(Problem,Population,Parent2,Parent3,{CR,F,1,20});
    [Population,FrontNo,CrowdDis] = EnvironmentalSelection_RLDMOEA([Population,Offspring],N);
end

%% =========================================================
function [Population,FrontNo,CrowdDis] = EnvironmentalSelection_RLDMOEA(Population,N)
    try
        [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
    catch
        [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    end
    Next = FrontNo < MaxFNo;

    CrowdDis = CrowdingDistance_RLDMOEA(Population.objs,FrontNo);

    Last = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;

    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
end

function CrowdDis = CrowdingDistance_RLDMOEA(PopObj,FrontNo)
    [N,M] = size(PopObj);
    CrowdDis = zeros(1,N);
    Fronts = setdiff(unique(FrontNo),inf);

    for f = 1 : length(Fronts)
        idx  = find(FrontNo==Fronts(f));
        if numel(idx) <= 2
            CrowdDis(idx) = inf;
            continue;
        end

        Fmax = max(PopObj(idx,:),[],1);
        Fmin = min(PopObj(idx,:),[],1);

        for m = 1 : M
            [~,rank] = sortrows(PopObj(idx,m));
            CrowdDis(idx(rank(1)))   = inf;
            CrowdDis(idx(rank(end))) = inf;

            if Fmax(m)-Fmin(m) < 1e-12
                continue;
            end

            for j = 2 : numel(idx)-1
                CrowdDis(idx(rank(j))) = CrowdDis(idx(rank(j))) + ...
                    (PopObj(idx(rank(j+1)),m)-PopObj(idx(rank(j-1)),m)) / (Fmax(m)-Fmin(m));
            end
        end
    end
end

%% =========================================================
function Archive = GetNonDominated_RLDMOEA(Population)
    if isempty(Population)
        Archive = Population;
        return;
    end
    try
        FrontNo = NDSort(Population.objs,Population.cons,1);
    catch
        FrontNo = NDSort(Population.objs,1);
    end
    Archive = Population(FrontNo==1);
end

function [kDec,kObj] = KneeSolution_RLDMOEA(Population)
    Obj = Population.objs;
    if size(Obj,1) == 1
        kDec = Population.decs;
        kObj = Population.objs;
        return;
    end

    % MMD-style approximation
    fmin = min(Obj,[],1);
    dist = sum(abs(Obj - repmat(fmin,size(Obj,1),1)),2);
    [~,id] = min(dist);

    kDec = Population(id).decs;
    kObj = Population(id).objs;
end

function reward = HypervolumeReward_RLDMOEA(PopObj)
    if isempty(PopObj)
        reward = 0;
        return;
    end

    PopObj = unique(PopObj,'rows');
    [N,M]  = size(PopObj);

    ideal = min(PopObj,[],1);
    ref   = max(PopObj,[],1) + 1e-6;

    if M == 2
        P  = sortrows(PopObj,1);
        hv = 0;
        y  = ref(2);
        for i = 1 : N
            if P(i,2) < y
                hv = hv + max(0,ref(1)-P(i,1))*max(0,y-P(i,2));
                y  = P(i,2);
            end
        end
        reward = hv / max(prod(ref-ideal),1e-12);
    else
        S      = 2000;
        sample = rand(S,M).*repmat(ref-ideal,S,1) + repmat(ideal,S,1);
        dom    = false(S,1);
        for i = 1 : N
            dom = dom | all(repmat(PopObj(i,:),S,1) <= sample,2);
        end
        reward = mean(dom);
    end
end

%% =========================================================
function tf = Dominates_RLDMOEA(obj1,obj2)
    tf = all(obj1<=obj2) && any(obj1<obj2);
end

function Cons = GetConsMatrix_RLDMOEA(Pop)
    try
        Cons = Pop.cons;
        if isempty(Cons)
            Cons = zeros(numel(Pop),0);
        end
    catch
        Cons = zeros(numel(Pop),0);
    end
end

function Dec = RandomDecs_RLDMOEA(N,lower,upper)
    D   = numel(lower);
    Dec = rand(N,D).*repmat(upper-lower,N,1) + repmat(lower,N,1);
end

function A = SampleRows_RLDMOEA(A0,n,replace)
    if n <= 0
        A = zeros(0,size(A0,2));
        return;
    end

    m = size(A0,1);
    if m == 0
        A = zeros(n,size(A0,2));
        return;
    end

    if replace || m < n
        idx = randi(m,n,1);
    else
        idx = randperm(m,n);
    end
    A = A0(idx,:);
end
classdef dCOEA < ALGORITHM
% <multi> <real> <dynamic>
% Dynamic competitive-cooperative coevolutionary algorithm

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            defaultSubN     = max(4,ceil(Problem.N/max(Problem.D,1)));
            defaultArchiveN = max(Problem.N,50);
            defaultDetect   = max(5,ceil(0.1*defaultArchiveN));
            defaultNComp    = min(5,max(2,Problem.D));
            defaultMemBatch = min(defaultArchiveN,max(Problem.M,5));
            defaultMemCap   = 2*defaultArchiveN;

            [subN,archiveN,compFreq,detectNum,nComp,gamma,memBatch,memCap] = ...
                Algorithm.ParameterSet(defaultSubN,defaultArchiveN,5,defaultDetect,defaultNComp,0.5,defaultMemBatch,defaultMemCap);

            %% Initialization of species
            D       = Problem.D;
            Species = cell(1,D);
            Rank    = cell(1,D);
            Niche   = cell(1,D);
            Reps    = zeros(1,D);

            for i = 1:D
                Species{i} = Problem.lower(i) + rand(subN,1).*(Problem.upper(i)-Problem.lower(i));
                Rank{i}    = inf(subN,1);
                Niche{i}   = inf(subN,1);
                Reps(i)    = Species{i}(randi(subN));
            end

            Archive    = SOLUTION.empty();
            Memory     = SOLUTION.empty();
            DisplayPop = SOLUTION.empty();
            gen        = 1;

            %% Initial cooperative cycle
            [Species,Reps,Rank,Niche,Archive,DisplayPop] = ...
                CooperativeProcess_dCOEA(Problem,Species,Reps,Archive,archiveN);
            Species = ReproduceSpecies_dCOEA(Species,Rank,Niche,Problem);

            %% Optimization
            while Algorithm.NotTerminated(DisplayPop)
                changed = DetectChange_dCOEA(Problem,Archive,detectNum);

                if changed
                    [Archive,Memory] = HandleChange_dCOEA(Problem,Archive,Memory,archiveN,memBatch,memCap);
                    [Species,Reps,Archive,DisplayPop] = ...
                        CompetitiveProcess_dCOEA(Problem,Species,Reps,Archive,archiveN,nComp,gamma,true);
                else
                    if mod(gen,compFreq) == 0
                        [Species,Reps,Archive,DisplayPop] = ...
                            CompetitiveProcess_dCOEA(Problem,Species,Reps,Archive,archiveN,nComp,gamma,true);
                    else
                        [Species,Reps,Rank,Niche,Archive,DisplayPop] = ...
                            CooperativeProcess_dCOEA(Problem,Species,Reps,Archive,archiveN);
                        Species = ReproduceSpecies_dCOEA(Species,Rank,Niche,Problem);
                    end
                end
                gen = gen + 1;
            end
        end
    end
end

%% ========================= Core Procedures =========================
function [Species,Reps,Rank,Niche,Archive,DisplayPop] = ...
    CooperativeProcess_dCOEA(Problem,Species,Reps,Archive,archiveN)

    D          = numel(Species);
    Rank       = cell(1,D);
    Niche      = cell(1,D);
    DisplayPop = SOLUTION.empty();

    for i = 1:D
        Xi = Species{i};
        ni = numel(Xi);
        Ri = inf(ni,1);
        Ni = inf(ni,1);

        for j = 1:ni
            dec    = Reps;
            dec(i) = Xi(j);
            sol    = Problem.Evaluation(dec);

            DisplayPop = [DisplayPop,sol];
            Archive    = UpdateArchive_dCOEA(Archive,sol,archiveN);
            [Ri(j),Ni(j)] = RankAndNiche_dCOEA(sol,Archive);
        end

        Rank{i}  = Ri;
        Niche{i} = Ni;
        best     = BestIndex_dCOEA(Ri,Ni);
        Reps(i)  = Xi(best);
    end

    if isempty(DisplayPop)
        DisplayPop = Archive;
    end
end

function [Species,Reps,Archive,DisplayPop] = ...
    CompetitiveProcess_dCOEA(Problem,Species,Reps,Archive,archiveN,nComp,gamma,allowStochastic)

    D          = numel(Species);
    DisplayPop = SOLUTION.empty();

    for i = 1:D
        vals    = Reps(i);
        srcType = 0;       % 0: incumbent rep, 1: other species, 2: stochastic
        srcID   = i;

        nRand  = 0;
        if allowStochastic
            nRand = max(0,round((nComp-1)*gamma));
        end
        nOther = max(0,nComp-1-nRand);

        others = setdiff(1:D,i);
        if ~isempty(others) && nOther > 0
            chosen = others(randperm(numel(others),min(numel(others),nOther)));
            while numel(chosen) < nOther
                chosen(end+1) = others(randi(numel(others)));
            end
            for s = chosen
                vals(end+1,1)    = TransferValue_dCOEA(Species{s}(randi(numel(Species{s}))),s,i,Problem);
                srcType(end+1,1) = 1;
                srcID(end+1,1)   = s;
            end
        end

        if allowStochastic && nRand > 0
            randVals = LatinScalarSamples_dCOEA(nRand,Problem.lower(i),Problem.upper(i));
            for r = 1:numel(randVals)
                vals(end+1,1)    = randVals(r);
                srcType(end+1,1) = 2;
                srcID(end+1,1)   = 0;
            end
        end

        R = inf(numel(vals),1);
        N = inf(numel(vals),1);
        for p = 1:numel(vals)
            dec    = Reps;
            dec(i) = vals(p);
            sol    = Problem.Evaluation(dec);

            DisplayPop = [DisplayPop,sol];
            Archive    = UpdateArchive_dCOEA(Archive,sol,archiveN);
            [R(p),N(p)] = RankAndNiche_dCOEA(sol,Archive);
        end

        win = BestIndex_dCOEA(R,N);

        if srcType(win) == 1
            s         = srcID(win);
            Species{i}= TransferSpecies_dCOEA(Species{s},s,i,Problem);
            Reps(i)   = TransferValue_dCOEA(Reps(s),s,i,Problem);
        elseif srcType(win) == 2
            Species{i}= RestartSpecies_dCOEA(numel(Species{i}),vals(win),i,Problem);
            Reps(i)   = vals(win);
        else
            Reps(i)   = vals(win);
        end

        Species{i} = RandomVariationSpecies_dCOEA(Species{i},i,Problem);
    end

    if isempty(DisplayPop)
        DisplayPop = Archive;
    end
end

function Species = ReproduceSpecies_dCOEA(Species,Rank,Niche,Problem)
    D = numel(Species);
    for i = 1:D
        X   = Species{i};
        R   = Rank{i};
        NC  = Niche{i};
        n   = numel(X);
        Off = zeros(n,1);

        for k = 1:2:n
            p1 = TournamentPick_dCOEA(R,NC);
            p2 = TournamentPick_dCOEA(R,NC);
            [c1,c2] = ScalarGA_dCOEA(X(p1),X(p2),Problem.lower(i),Problem.upper(i));
            Off(k) = c1;
            if k+1 <= n
                Off(k+1) = c2;
            end
        end
        Species{i} = Off;
    end
end

%% ========================= Dynamic Handling =========================
function changed = DetectChange_dCOEA(Problem,Archive,detectNum)
    changed = false;
    if isempty(Archive)
        return;
    end

    N = numel(Archive);
    k = min(detectNum,N);
    if k <= 0
        return;
    end

    idx    = randperm(N,k);
    oldObj = vertcat(Archive(idx).objs);
    oldCon = GetConsMatrix_dCOEA(Archive(idx));

    newPop = Problem.Evaluation(vertcat(Archive(idx).decs));
    newObj = vertcat(newPop.objs);
    newCon = GetConsMatrix_dCOEA(newPop);

    changed = ~isequal(oldObj,newObj) || ~isequal(oldCon,newCon);
end

function [Archive,Memory] = HandleChange_dCOEA(Problem,Archive,Memory,archiveN,memBatch,memCap)
    if isempty(Archive)
        return;
    end

    selected = SelectMemorySolutions_dCOEA(Archive,memBatch);
    if ~isempty(selected)
        Memory = [Memory,selected];
        if numel(Memory) > memCap
            Memory = Memory(end-memCap+1:end);
        end
    end

    Archive = SOLUTION.empty();

    if ~isempty(Memory)
        ReMem = Problem.Evaluation(vertcat(Memory.decs));
        Memory = ReMem;
        for i = 1:numel(ReMem)
            Archive = UpdateArchive_dCOEA(Archive,ReMem(i),archiveN);
        end
    end
end

%% ========================= Archive / Fitness =========================
function Archive = UpdateArchive_dCOEA(Archive,Candidate,archiveN)
    for t = 1:numel(Candidate)
        c = Candidate(t);

        if isempty(Archive)
            Archive = c;
            continue;
        end

        Aobj = vertcat(Archive.objs);
        cobj = c.obj;

        dominateByA = all(Aobj <= repmat(cobj,size(Aobj,1),1),2) & ...
                      any(Aobj <  repmat(cobj,size(Aobj,1),1),2);

        if any(dominateByA)
            continue;
        end

        dominateA = all(repmat(cobj,size(Aobj,1),1) <= Aobj,2) & ...
                    any(repmat(cobj,size(Aobj,1),1) <  Aobj,2);

        Archive = [Archive(~dominateA),c];
        Archive = RemoveDuplicate_dCOEA(Archive);

        while numel(Archive) > archiveN
            del = Truncation_dCOEA(vertcat(Archive.objs),1);
            Archive(del) = [];
        end
    end
end

function [rank,niche] = RankAndNiche_dCOEA(sol,Archive)
    if isempty(Archive)
        rank  = 0;
        niche = 0;
        return;
    end

    Aobj = vertcat(Archive.objs);
    xobj = sol.obj;

    rank = sum(all(Aobj <= repmat(xobj,size(Aobj,1),1),2) & ...
               any(Aobj <  repmat(xobj,size(Aobj,1),1),2));

    if size(Aobj,1) <= 1
        niche = 0;
        return;
    end

    DistAA = pdist2(Aobj,Aobj);
    DistAA(logical(eye(size(DistAA)))) = inf;
    sigma = mean(min(DistAA,[],2));

    if sigma <= 1e-12 || isnan(sigma)
        sigma = max(std(Aobj(:)),1e-6);
    end

    dist  = pdist2(xobj,Aobj);
    niche = sum(max(0,1 - dist./(sigma+eps)));
end

function Archive = RemoveDuplicate_dCOEA(Archive)
    if numel(Archive) <= 1
        return;
    end
    Obj = vertcat(Archive.objs);
    [~,ia,~] = unique(round(Obj*1e12)/1e12,'rows','stable');
    Archive = Archive(sort(ia));
end

function Del = Truncation_dCOEA(PopObj,K)
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(size(Distance)))) = inf;
    Del = false(1,size(PopObj,1));
    while sum(Del) < K
        Remain = find(~Del);
        Temp   = sort(Distance(Remain,Remain),2);
        [~,r]  = sortrows(Temp);
        Del(Remain(r(1))) = true;
    end
end

%% ========================= Memory Selection =========================
function selected = SelectMemorySolutions_dCOEA(Archive,memBatch)
    selected = SOLUTION.empty();
    if isempty(Archive) || memBatch <= 0
        return;
    end

    Aobj = vertcat(Archive.objs);
    n    = numel(Archive);
    M    = size(Aobj,2);

    idx = [];
    for m = 1:M
        [~,id] = min(Aobj(:,m));
        idx(end+1) = id;
    end
    idx = unique(idx,'stable');

    if numel(idx) > memBatch
        idx = idx(randperm(numel(idx),memBatch));
    end

    remain = setdiff(1:n,idx);
    while numel(idx) < memBatch && ~isempty(remain)
        p = remain(randi(numel(remain)));
        idx(end+1) = p;
        remain(remain==p) = [];
    end

    selected = Archive(idx);
end

%% ========================= Species Utilities =========================
function idx = BestIndex_dCOEA(R,N)
    [~,order] = sortrows([R(:),N(:)],[1 2]);
    idx = order(1);
end

function idx = TournamentPick_dCOEA(R,N)
    n = numel(R);
    if n == 1
        idx = 1;
        return;
    end
    a = randi(n);
    b = randi(n);
    if (R(a) < R(b)) || (R(a) == R(b) && N(a) <= N(b))
        idx = a;
    else
        idx = b;
    end
end

function [c1,c2] = ScalarGA_dCOEA(x1,x2,lb,ub)
    if rand < 0.9
        alpha = rand;
        c1 = alpha*x1 + (1-alpha)*x2;
        c2 = alpha*x2 + (1-alpha)*x1;
    else
        c1 = x1;
        c2 = x2;
    end

    if rand < 0.2
        c1 = c1 + 0.1*(ub-lb)*randn;
    end
    if rand < 0.2
        c2 = c2 + 0.1*(ub-lb)*randn;
    end

    c1 = min(max(c1,lb),ub);
    c2 = min(max(c2,lb),ub);
end

function Xnew = RandomVariationSpecies_dCOEA(X,dim,Problem)
    n    = numel(X);
    perm = randperm(n);
    X    = X(perm);
    Xnew = zeros(size(X));

    for k = 1:2:n
        if k == n
            c1 = X(k);
            if rand < 0.3
                c1 = c1 + 0.1*(Problem.upper(dim)-Problem.lower(dim))*randn;
            end
            Xnew(k) = min(max(c1,Problem.lower(dim)),Problem.upper(dim));
        else
            [c1,c2] = ScalarGA_dCOEA(X(k),X(k+1),Problem.lower(dim),Problem.upper(dim));
            Xnew(k)   = c1;
            Xnew(k+1) = c2;
        end
    end
end

function Xto = TransferSpecies_dCOEA(Xfrom,fromDim,toDim,Problem)
    lb1 = Problem.lower(fromDim);
    ub1 = Problem.upper(fromDim);
    lb2 = Problem.lower(toDim);
    ub2 = Problem.upper(toDim);

    Xnorm = (Xfrom - lb1) ./ (ub1 - lb1 + eps);
    Xto   = lb2 + Xnorm .* (ub2 - lb2);
    Xto   = min(max(Xto,lb2),ub2);
end

function vto = TransferValue_dCOEA(vfrom,fromDim,toDim,Problem)
    lb1 = Problem.lower(fromDim);
    ub1 = Problem.upper(fromDim);
    lb2 = Problem.lower(toDim);
    ub2 = Problem.upper(toDim);

    t   = (vfrom - lb1) / (ub1 - lb1 + eps);
    vto = lb2 + t*(ub2 - lb2);
    vto = min(max(vto,lb2),ub2);
end

function X = RestartSpecies_dCOEA(n,center,dim,Problem)
    lb   = Problem.lower(dim);
    ub   = Problem.upper(dim);
    span = 0.25*(ub-lb);
    X    = center + span*randn(n,1);
    X    = min(max(X,lb),ub);
end

function vals = LatinScalarSamples_dCOEA(n,lb,ub)
    u    = ((0:n-1)' + rand(n,1)) / n;
    u    = u(randperm(n));
    vals = lb + u*(ub-lb);
end

%% ========================= Constraint Utility =========================
function C = GetConsMatrix_dCOEA(Pop)
    if isempty(Pop)
        C = [];
        return;
    end
    try
        C = vertcat(Pop.cons);
    catch
        C = [];
    end
end
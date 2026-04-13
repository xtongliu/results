function Population = ApplyResponseStrategy_D(Problem, Population, actionID, history, severity)
% ApplyResponseStrategy_D
% -------------------------------------------------------------------------
% Agent 1 response module redesigned with an SGEA-style recovery skeleton.
%
% Agent 1 does NOT choose among unrelated response paradigms any more.
% Instead, all actions follow the same "retain + directed migration + noise"
% logic, and actions only control the response intensity.
%
% actionID:
%   1 - Mild recovery
%   2 - Standard recovery
%   3 - Strong recovery
%   4 - Recovery with partial restart
% -------------------------------------------------------------------------

    if nargin < 5 || isempty(severity)
        severity = 0.2;
    end
    severity = min(max(severity, 0), 1);

    N = length(Population);
    D = Problem.D;
    lower = Problem.lower;
    upper = Problem.upper;
    span  = upper - lower;
    span(span < 1e-12) = 1;

    % Action-dependent response intensity
    switch actionID
        case 1
            remainRatio  = 0.70;
            stepFactor   = 0.45;
            noiseFactor  = 0.020;
            restartRatio = 0.00;
        case 2
            remainRatio  = 0.50;
            stepFactor   = 0.85;
            noiseFactor  = 0.035;
            restartRatio = 0.00;
        case 3
            remainRatio  = 0.30;
            stepFactor   = 1.25;
            noiseFactor  = 0.060;
            restartRatio = 0.00;
        case 4
            remainRatio  = 0.20;
            stepFactor   = 1.00;
            noiseFactor  = 0.050;
            restartRatio = 0.30;
        otherwise
            remainRatio  = 0.50;
            stepFactor   = 0.85;
            noiseFactor  = 0.035;
            restartRatio = 0.00;
    end

    remainNum = max(2, min(N, round(remainRatio * N)));
    keepMask  = localDiverseRemain(Population.objs, remainNum);
    remainPop = Population(keepMask);

    if isempty(remainPop)
        remainPop = Population(localSelectBestIndices(Population, max(2, ceil(0.2 * N))));
    end

    genNum     = N - length(remainPop);
    restartNum = round(restartRatio * N);
    restartNum = min(restartNum, genNum);
    shiftNum   = genNum - restartNum;

    % ---------------------------------------------------------------------
    % Build SGEA-style directed recovery
    % ---------------------------------------------------------------------
    remainDecs = remainPop.decs;
    CR = mean(remainDecs, 1);

    % Historical archive center
    if history.count >= 1 && ~isempty(history.decs{history.count})
        histDecs = history.decs{history.count};
        CA = mean(histDecs, 1);
    else
        histDecs = remainDecs;
        CA = CR;
    end

    % Historical drift
    if history.count >= 2
        drift = history.decCenters{history.count} - history.decCenters{history.count-1};
    else
        drift = zeros(1, D);
    end

    % Main transfer direction
    dir1 = CA - CR;
    dir2 = drift;
    if norm(dir1) < 1e-12
        dir1 = zeros(1, D);
    else
        dir1 = dir1 ./ (norm(dir1) + 1e-12);
    end
    if norm(dir2) < 1e-12
        dir2 = zeros(1, D);
    else
        dir2 = dir2 ./ (norm(dir2) + 1e-12);
    end

    direction = 0.65 * dir1 + 0.35 * dir2;
    if norm(direction) < 1e-12
        direction = randn(1, D);
    end
    direction = direction ./ (norm(direction) + 1e-12);

    stepScale = stepFactor * (0.08 + 0.42 * severity);
    moveVec   = stepScale * mean(span) * direction;

    if shiftNum > 0
        sourceDecs = localMixSources(remainDecs, histDecs, shiftNum);
        noiseSigma = noiseFactor * (0.6 + severity) .* span;
        shiftDecs  = sourceDecs + repmat(moveVec, shiftNum, 1) + ...
                     randn(shiftNum, D) .* repmat(noiseSigma, shiftNum, 1);
        shiftDecs  = localBoundRepair(shiftDecs, lower, upper);
        newPop1    = Problem.Evaluation(shiftDecs);
    else
        newPop1 = [];
    end

    if restartNum > 0
        restartDecs = localRandomDecs(restartNum, lower, upper);
        newPop2 = Problem.Evaluation(restartDecs);
    else
        newPop2 = [];
    end

    Population = [remainPop, newPop1, newPop2];
    Population = localTruncatePopulation(Population, N);
end

%% ========================= Local helper functions ========================

function keepMask = localDiverseRemain(PopObj, remainNum)
    N = size(PopObj, 1);
    delNum = max(0, N - remainNum);
    if delNum == 0
        keepMask = true(1, N);
        return;
    end
    delMask = localTruncation(PopObj, delNum);
    keepMask = ~delMask;
end

function Del = localTruncation(PopObj, K)
    Distance = pdist2(PopObj, PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1, size(PopObj, 1));
    while sum(Del) < K
        Remain = find(~Del);
        Temp   = sort(Distance(Remain, Remain), 2);
        [~, Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end

function idx = localSelectBestIndices(Population, K)
    FrontNo  = NDSort(Population.objs, inf);
    CrowdDis = CrowdingDistance(Population.objs, FrontNo);
    [~, rank] = sortrows([FrontNo', -CrowdDis'], [1, 2]);
    idx = rank(1:min(K, numel(rank)));
end

function X = localMixSources(A, B, n)
    if isempty(A) && isempty(B)
        X = [];
        return;
    elseif isempty(A)
        X = localSampleRows(B, n);
        return;
    elseif isempty(B)
        X = localSampleRows(A, n);
        return;
    end

    nA = round(0.5 * n);
    nB = n - nA;
    XA = localSampleRows(A, nA);
    XB = localSampleRows(B, nB);
    X  = [XA; XB];
    X  = X(randperm(size(X, 1)), :);
end

function X = localSampleRows(A, n)
    if isempty(A)
        X = [];
        return;
    end
    m = size(A, 1);
    idx = randi(m, n, 1);
    X = A(idx, :);
end

function X = localRandomDecs(n, lower, upper)
    D = numel(lower);
    X = rand(n, D) .* repmat(upper - lower, n, 1) + repmat(lower, n, 1);
end

function X = localBoundRepair(X, lower, upper)
    X = min(max(X, repmat(lower, size(X, 1), 1)), repmat(upper, size(X, 1), 1));
end

function Population = localTruncatePopulation(Population, N)
    if length(Population) <= N
        return;
    end

    PopObj = Population.objs;
    [FrontNo, MaxFNo] = NDSort(PopObj, N);
    Next = FrontNo < MaxFNo;

    Last = find(FrontNo == MaxFNo);
    CrowdDis = CrowdingDistance(PopObj, FrontNo);
    [~, rank] = sort(CrowdDis(Last), 'descend');
    Last = Last(rank(1:(N - sum(Next))));

    Next(Last) = true;
    Population = Population(Next);
end

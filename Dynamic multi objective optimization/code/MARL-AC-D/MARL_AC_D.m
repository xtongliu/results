classdef MARL_AC_D < ALGORITHM
% <multi/many> <real> <dynamic>
% Multi-agent reinforcement learning based automated configuration for
% dynamic multi-objective optimization

    methods
        function main(Algorithm, Problem)
            %% Parameter setting
            [detectNum, memorySize, respWindow] = Algorithm.ParameterSet(5, 5, 8);

            %% MOEA/D initialization
            [W, Problem.N] = UniformPoint(Problem.N, Problem.M);
            T = ceil(Problem.N / 10);
            T = max(T, 2);

            Distance = pdist2(W, W);
            [~, B] = sort(Distance, 2);
            B = B(:, 1:T);

            Population = Problem.Initialization();
            Z = min(Population.objs, [], 1);

            AllPop = [];

            %% MARL agent initialization
            agent = MAAGENTDMOP(3, [6, 6, 5], [4, 4, 4], [1, 1, 1], {[], [], []});

            %% History memory
            history = initHistory(memorySize);

            %% Context initialization
            ctx = struct();
            ctx.gensSinceChange   = respWindow + 1;
            ctx.responseAction    = 2;
            ctx.searchAction      = 2;
            ctx.responseHorizon   = respWindow;
            ctx.stagnationCounter = 0;
            ctx.stagnationWindow  = 10;
            ctx.refPF             = [];
            ctx.changeSeverity    = 0;
            ctx.changeDetected    = false;
            ctx.historyCount      = 0;
            ctx.historyDrift      = 0;
            ctx.detectAction      = 2;
            ctx.detectMean        = 0;
            ctx.detectStd         = 0;
            ctx.detectCost        = min(detectNum / max(Problem.N, 1), 1);
            ctx.detectRawSeverity = 0;
            ctx.detectTriggered   = false;
            ctx.falseAlarmProxy   = 0;

            [state, ~, ~] = BuildDynamicMARLSignals(Problem, Population, Population, ctx);

            %% Runtime variables
            lastResponseAction = 2;
            lastSearchAction   = 2;
            lastDetectAction   = 2;
            stagnationCounter  = 0;
            gensSinceChange    = respWindow + 1;
            lastSeverity       = 0;
            detectInfo = struct('meanRelDiff', 0, 'stdRelDiff', 0, 'evalRatio', min(detectNum / max(Problem.N, 1), 1), ...
                'triggered', false, 'falseAlarmProxy', 0, 'epsilon', 1e-4, 'sentinelNum', detectNum, 'rawSeverity', 0);
            warmupRatio = 0.15;

            %% Optimization loop
            while Algorithm.NotTerminated(Population)
                oldPop = Population;

                % =========================================================
                % 1) Detect environmental change
                % =========================================================
                rawAction = agent.Action(state);
                allActions = agent.SelectAllActions(rawAction, Problem);
                detectAction = allActions(3);

                if Problem.FE < warmupRatio * Problem.maxFE
                    detectAction = 2;
                end

                detectCfg = localBuildDetectConfig(detectAction, detectNum, Problem, lastSeverity);
                [changeDetected, reevalPop, severity, detectInfo] = DetectChange_D(Problem, Population, detectCfg);

                if changeDetected
                    % Store previous-environment knowledge
                    AllPop = [AllPop, Population];
                    history = updateHistory(history, oldPop);

                    % Reevaluate current population in new environment
                    Population = reevalPop;
                    Z = min(Population.objs, [], 1);

                    gensSinceChange = 0;
                    lastSeverity = severity;

                    % Build state right after the environment change
                    ctx.gensSinceChange   = 0;
                    ctx.responseAction    = lastResponseAction;
                    ctx.searchAction      = lastSearchAction;
                    ctx.responseHorizon   = respWindow;
                    ctx.stagnationCounter = stagnationCounter;
                    ctx.stagnationWindow  = 10;
                    ctx.changeSeverity    = severity;
                    ctx.changeDetected    = true;
                    ctx.historyCount      = history.count;
                    ctx.historyDrift      = computeHistoryDrift(history, Problem);
                    ctx.detectAction      = detectAction;
                    ctx.detectMean        = detectInfo.meanRelDiff;
                    ctx.detectStd         = detectInfo.stdRelDiff;
                    ctx.detectCost        = detectInfo.evalRatio;
                    ctx.detectRawSeverity = detectInfo.rawSeverity;
                    ctx.detectTriggered   = true;
                    ctx.falseAlarmProxy   = 0;

                    [state, ~, ~] = BuildDynamicMARLSignals(Problem, oldPop, Population, ctx);

                    % Joint decision at the change point
                    rawAction = agent.Action(state);
                    allActions = agent.SelectAllActions(rawAction, Problem);
                    responseAction = allActions(1);
                    searchAction   = allActions(2);

                    % Agent 1 executes the response strategy
                    Population = ApplyResponseStrategy_D(Problem, Population, responseAction, history, severity);
                    Z = min(Population.objs, [], 1);

                    lastResponseAction = responseAction;
                    lastSearchAction   = searchAction;
                    lastDetectAction   = detectAction;
                    % For reward consistency: use reevaluated previous population
                    % in the current environment as HV_{t-1}.
                    transitionStartPop = reevalPop;
                else
                    % No new change: Agent 1 holds the latest response action.
                    lastSearchAction = allActions(2);
                    lastDetectAction = detectAction;
                    transitionStartPop = Population;
                end

                % =========================================================
                % 2) One-generation search under Agent 2 search mode
                % =========================================================
                [Population, Z] = SearchOneGeneration_D( ...
                    Problem, Population, W, B, T, Z, lastSearchAction, history, gensSinceChange);

                % =========================================================
                % 3) Update counters and context
                % =========================================================
                gensSinceChange = gensSinceChange + 1;

                ctx.gensSinceChange   = gensSinceChange;
                ctx.responseAction    = lastResponseAction;
                ctx.searchAction      = lastSearchAction;
                ctx.responseHorizon   = respWindow;
                ctx.stagnationCounter = stagnationCounter;
                ctx.stagnationWindow  = 10;
                ctx.changeSeverity    = lastSeverity;
                ctx.changeDetected    = changeDetected;
                ctx.historyCount      = history.count;
                ctx.historyDrift      = computeHistoryDrift(history, Problem);
                ctx.detectAction      = lastDetectAction;
                ctx.detectMean        = detectInfo.meanRelDiff;
                ctx.detectStd         = detectInfo.stdRelDiff;
                ctx.detectCost        = detectInfo.evalRatio;
                ctx.detectRawSeverity = detectInfo.rawSeverity;
                ctx.detectTriggered   = detectInfo.triggered;
                ctx.falseAlarmProxy   = detectInfo.falseAlarmProxy;

                % =========================================================
                % 4) Next state and reward
                % =========================================================
                [nextState, rewardVec, metricInfo] = ...
                    BuildDynamicMARLSignals(Problem, transitionStartPop, Population, ctx);

                if abs(metricInfo.hvGain) < 1e-4
                    stagnationCounter = stagnationCounter + 1;
                else
                    stagnationCounter = 0;
                end

                if detectInfo.triggered
                    detectInfo.falseAlarmProxy = double(abs(metricInfo.hvGain) < 1e-4);
                else
                    detectInfo.falseAlarmProxy = 0;
                end

                % =========================================================
                % 5) Store discrete experience and train
                % =========================================================
                actionIdx = [lastResponseAction; lastSearchAction; lastDetectAction];
                agent.Experience(state, actionIdx, rewardVec, nextState);
                agent.Train();

                % =========================================================
                % 6) Move to next state
                % =========================================================
                state = nextState;

                if Problem.FE >= Problem.maxFE
                    Population = [AllPop, Population];
                    [~, rank] = sort(Population.adds(zeros(length(Population), 1)));
                    Population = Population(rank);
                end
            end
        end
    end
end

%% ========================================================================
% History memory
% ========================================================================

function history = initHistory(memorySize)
    history.maxSize    = memorySize;
    history.count      = 0;
    history.decs       = cell(1, memorySize);
    history.objs       = cell(1, memorySize);
    history.decCenters = cell(1, memorySize);
    history.objCenters = cell(1, memorySize);
end

function history = updateHistory(history, Population)
    FrontNo = NDSort(Population.objs, 1);
    ndMask  = (FrontNo == 1);

    ndDecs = Population(ndMask).decs;
    ndObjs = Population(ndMask).objs;

    if isempty(ndDecs)
        ndDecs = Population.decs;
        ndObjs = Population.objs;
    end

    decCenter = mean(ndDecs, 1);
    objCenter = mean(ndObjs, 1);

    if history.count < history.maxSize
        history.count = history.count + 1;
        pos = history.count;
    else
        history.decs(1:end-1)       = history.decs(2:end);
        history.objs(1:end-1)       = history.objs(2:end);
        history.decCenters(1:end-1) = history.decCenters(2:end);
        history.objCenters(1:end-1) = history.objCenters(2:end);
        pos = history.maxSize;
    end

    history.decs{pos}       = ndDecs;
    history.objs{pos}       = ndObjs;
    history.decCenters{pos} = decCenter;
    history.objCenters{pos} = objCenter;
end

function drift = computeHistoryDrift(history, Problem)
    if history.count < 2
        drift = 0;
        return;
    end

    d = history.decCenters{history.count} - history.decCenters{history.count - 1};
    denom = norm(Problem.upper - Problem.lower) + 1e-12;
    drift = min(norm(d) / denom, 1);
end

function cfg = localBuildDetectConfig(actionID, baseDetectNum, Problem, prevSeverity)
    if nargin < 4
        prevSeverity = 0;
    end

    baseNum = max(2, min(Problem.N, round(baseDetectNum)));

    switch actionID
        case 1
            % few sentinels + loose threshold + single-shot
            cfg.detectNum = max(2, min(Problem.N, round(0.60 * baseNum)));
            cfg.epsilon = 3e-4;
            cfg.smoothAlpha = 0.0;
            cfg.confirmMode = 1;
            cfg.confirmNum  = 0;
            cfg.confirmEpsilonScale = 1.00;
        case 2
            % medium sentinels + medium threshold + smoothed confirmation
            cfg.detectNum = max(3, min(Problem.N, baseNum));
            cfg.epsilon = 1e-4;
            cfg.smoothAlpha = 0.35;
            cfg.confirmMode = 2;
            cfg.confirmNum  = 0;
            cfg.confirmEpsilonScale = 1.00;
        case 3
            % many sentinels + strict threshold + single-shot
            cfg.detectNum = max(4, min(Problem.N, round(1.60 * baseNum)));
            cfg.epsilon = 5e-5;
            cfg.smoothAlpha = 0.0;
            cfg.confirmMode = 1;
            cfg.confirmNum  = 0;
            cfg.confirmEpsilonScale = 1.00;
        case 4
            % medium-high sentinels + medium threshold + two-stage confirmation
            cfg.detectNum = max(4, min(Problem.N, round(1.20 * baseNum)));
            cfg.epsilon = 1e-4;
            cfg.smoothAlpha = 0.25;
            cfg.confirmMode = 3;
            cfg.confirmNum  = max(3, min(Problem.N, round(0.80 * cfg.detectNum)));
            cfg.confirmEpsilonScale = 1.10;
        otherwise
            cfg.detectNum = baseNum;
            cfg.epsilon = 1e-4;
            cfg.smoothAlpha = 0.0;
            cfg.confirmMode = 1;
            cfg.confirmNum  = 0;
            cfg.confirmEpsilonScale = 1.00;
    end

    cfg.prevSeverity = min(max(prevSeverity, 0), 1);
    cfg.actionID = actionID;
end

function v = localClipScalar(v, lb, ub)
    v = min(max(v, lb), ub);
end

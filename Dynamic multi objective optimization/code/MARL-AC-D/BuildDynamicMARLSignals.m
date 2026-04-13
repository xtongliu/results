function [jointState, rewardVec, metricInfo] = BuildDynamicMARLSignals(Problem, prevPop, currPop, ctx)
% BuildDynamicMARLSignals
% -------------------------------------------------------------------------
% State and reward construction for the redesigned three-agent dynamic MARL.
%
% Agent 1: change-response intensity selector
% Agent 2: post-change tracking-search scheduler
% Agent 3: change-detection strategy selector
% -------------------------------------------------------------------------

    if nargin < 4 || isempty(ctx)
        ctx = struct();
    end

    % Defaults
    if ~isfield(ctx, 'gensSinceChange'),   ctx.gensSinceChange   = 0;  end
    if ~isfield(ctx, 'responseAction'),    ctx.responseAction    = 1;  end
    if ~isfield(ctx, 'searchAction'),      ctx.searchAction      = 1;  end
    if ~isfield(ctx, 'responseHorizon'),   ctx.responseHorizon   = 8;  end
    if ~isfield(ctx, 'stagnationCounter'), ctx.stagnationCounter = 0;  end
    if ~isfield(ctx, 'stagnationWindow'),  ctx.stagnationWindow  = 10; end
    if ~isfield(ctx, 'refPF'),             ctx.refPF             = []; end
    if ~isfield(ctx, 'changeSeverity'),    ctx.changeSeverity    = 0;  end
    if ~isfield(ctx, 'changeDetected'),    ctx.changeDetected    = false; end
    if ~isfield(ctx, 'historyCount'),      ctx.historyCount      = 0;  end
    if ~isfield(ctx, 'historyDrift'),      ctx.historyDrift      = 0;  end
    if ~isfield(ctx, 'detectAction'),      ctx.detectAction      = 1;  end
    if ~isfield(ctx, 'detectMean'),        ctx.detectMean        = 0;  end
    if ~isfield(ctx, 'detectStd'),         ctx.detectStd         = 0;  end
    if ~isfield(ctx, 'detectCost'),        ctx.detectCost        = 0;  end
    if ~isfield(ctx, 'detectRawSeverity'), ctx.detectRawSeverity = 0;  end
    if ~isfield(ctx, 'detectTriggered'),   ctx.detectTriggered   = false; end
    if ~isfield(ctx, 'falseAlarmProxy'),   ctx.falseAlarmProxy   = 0;  end

    progressRatio   = Problem.FE / max(Problem.maxFE, 1);
    changeAgeRatio  = min(ctx.gensSinceChange / max(ctx.responseHorizon, 1), 1);
    changeWinRatio  = max(0, 1 - changeAgeRatio);
    stagnationRatio = min(ctx.stagnationCounter / max(ctx.stagnationWindow, 1), 1);

    % Use consistent normalization across prev/curr populations
    allObj = [prevPop.objs; currPop.objs];
    objMin = min(allObj, [], 1);
    objMax = max(allObj, [], 1);
    objRange = objMax - objMin;
    objRange(objRange < 1e-12) = 1;

    decLower = Problem.lower;
    decUpper = Problem.upper;
    decRange = decUpper - decLower;
    decRange(decRange < 1e-12) = 1;

    prevInfo = localProfile(prevPop, objMin, objRange, decLower, decRange);
    currInfo = localProfile(currPop, objMin, objRange, decLower, decRange);

    objShift = localRelChange(currInfo.objCenter, prevInfo.objCenter);
    decShift = localRelChange(currInfo.decCenter, prevInfo.decCenter);
    ndGain   = localClip(currInfo.ndRatio - prevInfo.ndRatio, -1, 1);

    refPoint = localBuildHVReference(allObj);
    hvPrev   = HV(prevPop, refPoint);
    hvCurr   = HV(currPop, refPoint);
    hvGain   = localSafeGain(hvPrev, hvCurr);

    if ~isempty(ctx.refPF)
        prevConv = IGD(prevPop, ctx.refPF);
        currConv = IGD(currPop, ctx.refPF);
        convGain = localSafeGain(currConv, prevConv);  % smaller is better
        convMode = 1;
    else
        prevConv = prevInfo.convProxy;
        currConv = currInfo.convProxy;
        convGain = localSafeGain(currConv, prevConv);  % smaller is better
        convMode = 0;
    end

    divGain = localSafeGain(prevInfo.objSpread, currInfo.objSpread);
    responseScore = localClip(0.40 * convGain + 0.30 * hvGain + 0.20 * ndGain + 0.10 * divGain, -1, 1);

    % Response/search/detection states with shared basic perception
    % (progress, change age, change severity) and task-specific features.
    currConvLevel = localClip(1 - currConv, 0, 1);
    currDivLevel  = localNormalizeSpread(currInfo.objSpread, size(currPop.objs, 2));

    agent1State = [
        progressRatio;                             % 1: FE utilization ratio
        changeAgeRatio;                            % 2: generations since last detected change
        min(max(ctx.changeSeverity, 0), 1);       % 3: detected change severity
        objShift;                                  % 4: objective-space shift
        decShift;                                  % 5: decision-space shift
        min(max(ctx.historyDrift, 0), 1)          % 6: history-drift indicator
    ];

    agent2State = [
        progressRatio;                             % 1: FE utilization ratio
        changeAgeRatio;                            % 2: generations since last detected change
        min(max(ctx.changeSeverity, 0), 1);       % 3: detected change severity
        currInfo.ndRatio;                          % 4: nondominated ratio level
        currConvLevel;                             % 5: convergence level
        currDivLevel                               % 6: diversity level
    ];

    % Detection state: stage + change context + sentinel-signal statistics.
    agent3State = [
        progressRatio;                             % 1: FE utilization ratio
        changeAgeRatio;                            % 2: generations since last detected change
        min(max(ctx.changeSeverity, 0), 1);       % 3: detected change severity
        localClip(ctx.detectStd, 0, 1);           % 4: variation dispersion
        localClip(ctx.detectCost, 0, 1)           % 5: detection cost ratio
    ];

    jointState = [agent1State; agent2State; agent3State];

    % Global shared reward:
    % r_t = (HV_t - HV_{t-1}) / (|HV_{t-1}| + eps)
    % The caller ensures HV_t and HV_{t-1} are compared in the same environment.
    globalReward = localClip(hvGain, -1, 1);
    rewardVec = [globalReward; globalReward; globalReward];

    metricInfo = struct();
    metricInfo.hvPrev        = hvPrev;
    metricInfo.hvCurr        = hvCurr;
    metricInfo.hvGain        = hvGain;
    metricInfo.convPrev      = prevConv;
    metricInfo.convCurr      = currConv;
    metricInfo.convGain      = convGain;
    metricInfo.divGain       = divGain;
    metricInfo.ndGain        = ndGain;
    metricInfo.objShift      = objShift;
    metricInfo.decShift      = decShift;
    metricInfo.responseScore = responseScore;
    metricInfo.globalReward  = globalReward;
    metricInfo.convMode      = convMode;
    metricInfo.agent1State   = agent1State;
    metricInfo.agent2State   = agent2State;
    metricInfo.agent3State   = agent3State;
    metricInfo.detectMean    = ctx.detectMean;
    metricInfo.detectStd     = ctx.detectStd;
    metricInfo.detectCost    = ctx.detectCost;
    metricInfo.detectRawSeverity = ctx.detectRawSeverity;
    metricInfo.falseAlarmProxy = ctx.falseAlarmProxy;
end

%% ========================================================================
% Helper functions
% ========================================================================

function info = localProfile(pop, objMin, objRange, decLower, decRange)
    objs = pop.objs;
    decs = pop.decs;
    N = size(objs, 1);
    M = size(objs, 2);

    normObj = (objs - repmat(objMin, N, 1)) ./ repmat(objRange, N, 1);
    normDec = (decs - repmat(decLower, N, 1)) ./ repmat(decRange, N, 1);

    info.objCenter = mean(normObj, 1);
    info.decCenter = mean(normDec, 1);
    info.objSpread = mean(sqrt(sum((normObj - repmat(info.objCenter, N, 1)).^2, 2)));
    info.decSpread = mean(sqrt(sum((normDec - repmat(info.decCenter, N, 1)).^2, 2)));

    frontNo = NDSort(objs, 1);
    info.ndRatio = sum(frontNo == 1) / max(N, 1);

    ideal = min(normObj, [], 1);
    info.convProxy = mean(sqrt(sum((normObj - repmat(ideal, N, 1)).^2, 2))) / max(sqrt(M), 1);
end

function v = localRelChange(xNew, xOld)
    denom = norm(xOld) + 1e-12;
    v = norm(xNew - xOld) / denom;
    v = min(v, 5) / 5;
end

function g = localSafeGain(oldVal, newVal)
    g = (newVal - oldVal) / (abs(oldVal) + 1e-12);
    g = localClip(g, -1, 1);
end

function x = localNormalizeAction(actionID, maxAction)
    x = (actionID - 1) / max(maxAction - 1, 1);
end

function y = localNormalizeSpread(spreadVal, dim)
    y = min(max(spreadVal / max(sqrt(dim), 1), 0), 1);
end

function x = localClip(x, lb, ub)
    x = min(max(x, lb), ub);
end

function refPoint = localBuildHVReference(PopObj)
    objMax = max(PopObj, [], 1);
    objMin = min(PopObj, [], 1);
    objRange = objMax - objMin;
    objRange(objRange < 1e-12) = 1;
    refPoint = objMax + 0.1 * objRange;
end

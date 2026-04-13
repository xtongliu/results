function [changed, reevaluatedPop, severity, detectInfo] = DetectChange_D(Problem, Population, detectCfg)
% DetectChange_D
% -------------------------------------------------------------------------
% Dynamic environment change detection by sentinel reevaluation.
%
% Output:
%   changed        - whether a change is detected
%   reevaluatedPop - reevaluated whole population if changed
%   severity       - relative change severity in [0,1]
%   detectInfo     - detection statistics used by MARL state/reward
% -------------------------------------------------------------------------

    if isnumeric(detectCfg)
        cfg.detectNum = detectCfg;
        cfg.epsilon = 1e-4;
        cfg.smoothAlpha = 0.0;
        cfg.prevSeverity = 0;
        cfg.actionID = 2;
        cfg.confirmMode = 1;
        cfg.confirmNum = 0;
        cfg.confirmEpsilonScale = 1.0;
    else
        cfg = detectCfg;
    end

    if ~isfield(cfg, 'detectNum'),    cfg.detectNum = 5; end
    if ~isfield(cfg, 'epsilon'),      cfg.epsilon = 1e-4; end
    if ~isfield(cfg, 'smoothAlpha'),  cfg.smoothAlpha = 0.0; end
    if ~isfield(cfg, 'prevSeverity'), cfg.prevSeverity = 0; end
    if ~isfield(cfg, 'actionID'),     cfg.actionID = 2; end
    if ~isfield(cfg, 'confirmMode'),  cfg.confirmMode = 1; end
    if ~isfield(cfg, 'confirmNum'),   cfg.confirmNum = 0; end
    if ~isfield(cfg, 'confirmEpsilonScale'), cfg.confirmEpsilonScale = 1.0; end

    N = length(Population);
    k = min(max(1, round(cfg.detectNum)), N);

    if k <= 0
        changed = false;
        reevaluatedPop = Population;
        severity = 0;
        detectInfo = localBuildDetectInfo(0, 0, 0, 0, k, 0, cfg, N, changed, false, false);
        return;
    end

    [rawSeverity1, meanDiff1, stdDiff1] = localEstimateSeverity(Problem, Population, k);
    rawSeverity = rawSeverity1;
    meanDiff = meanDiff1;
    stdDiff  = stdDiff1;

    alpha = min(max(cfg.smoothAlpha, 0), 1);
    severity = (1 - alpha) * rawSeverity + alpha * min(max(cfg.prevSeverity, 0), 1);
    severity = min(max(severity, 0), 1);

    eps0 = max(cfg.epsilon, 0);
    preTrigger = severity > eps0;
    changed = preTrigger;

    secondPassUsed = false;
    secondPassTriggered = false;
    k2 = 0;

    if cfg.confirmMode == 3 && preTrigger
        secondPassUsed = true;
        k2 = min(max(1, round(cfg.confirmNum)), N);
        [rawSeverity2, meanDiff2, stdDiff2] = localEstimateSeverity(Problem, Population, k2);

        eps2 = eps0 * max(cfg.confirmEpsilonScale, 1e-6);
        secondPassTriggered = rawSeverity2 > eps2;
        changed = preTrigger && secondPassTriggered;

        % Blend two passes so severity reflects both trigger and confirmation.
        rawSeverity = min(max(0.5 * rawSeverity1 + 0.5 * rawSeverity2, 0), 1);
        meanDiff = min(max(0.5 * meanDiff1 + 0.5 * meanDiff2, 0), 1);
        stdDiff  = min(max(0.5 * stdDiff1 + 0.5 * stdDiff2, 0), 1);
        severity = min(max((1 - alpha) * rawSeverity + alpha * min(max(cfg.prevSeverity, 0), 1), 0), 1);
    end

    evalRatio = min((k + k2) / max(N, 1), 1);
    detectInfo = localBuildDetectInfo(rawSeverity, meanDiff, stdDiff, severity, k, k2, cfg, N, changed, secondPassUsed, secondPassTriggered);

    % False alarm proxy is refined later by search performance feedback.
    detectInfo.falseAlarmProxy = 0;

    if changed
        reevaluatedPop = Problem.Evaluation(Population.decs);
    else
        reevaluatedPop = Population;
    end

    detectInfo.evalRatio = evalRatio;
end

function detectInfo = localBuildDetectInfo(rawSeverity, meanDiff, stdDiff, severity, k, k2, cfg, N, changed, secondPassUsed, secondPassTriggered)
    detectInfo = struct();
    detectInfo.rawSeverity = min(max(rawSeverity, 0), 1);
    detectInfo.meanRelDiff = min(max(meanDiff, 0), 1);
    detectInfo.stdRelDiff = min(max(stdDiff, 0), 1);
    detectInfo.severity = min(max(severity, 0), 1);
    detectInfo.sentinelNum = k;
    detectInfo.confirmSentinelNum = k2;
    detectInfo.totalSentinelNum = k + k2;
    detectInfo.epsilon = cfg.epsilon;
    detectInfo.actionID = cfg.actionID;
    detectInfo.confirmMode = cfg.confirmMode;
    detectInfo.secondPassUsed = secondPassUsed;
    detectInfo.secondPassTriggered = secondPassTriggered;
    detectInfo.evalRatio = min((k + k2) / max(N, 1), 1);
    detectInfo.triggered = changed;
    detectInfo.falseAlarmProxy = 0;
end

function [rawSeverity, meanDiff, stdDiff] = localEstimateSeverity(Problem, Population, k)
    N = length(Population);
    kk = min(max(1, round(k)), N);

    idx = randperm(N, kk);
    oldObjs = Population(idx).objs;
    decs    = Population(idx).decs;

    newPart = Problem.Evaluation(decs);
    newObjs = newPart.objs;

    relDiff = abs(newObjs - oldObjs) ./ (abs(oldObjs) + 1e-8);
    rawSeverity = mean(relDiff(:));
    rawSeverity = min(max(rawSeverity, 0), 1);
    meanDiff = min(max(mean(relDiff(:)), 0), 1);
    stdDiff = min(max(std(relDiff(:)), 0), 1);
end

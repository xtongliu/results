function [LastState, CurrentState, reward] = CalStateReward(Problem, LastPopulation, LastMask, Population, Mask, LastFE)
    % ==========================================================

    % ==========================================================
    LastPopulationHV = HV(LastPopulation, max([LastPopulation.objs; Population.objs]));
    PopulationHV     = HV(Population, max([LastPopulation.objs; Population.objs]));
    reward  = (PopulationHV - LastPopulationHV) / (LastPopulationHV + 1e-10);

    Lastsearch_stage = LastFE / Problem.maxFE;

    % ==========================================================

    % ==========================================================
    % ----------------------------------------------------------
    % Agent 1: Fitness selection (State_gv - Last)
    % ----------------------------------------------------------
    LastSparse = mean(LastMask(:));
    LastStd    = std(sum(LastMask,2))/size(LastMask,2);
    LastSparseDistribution = histcounts(sum(LastMask,2)./size(LastMask,2), 0:0.1:1)./size(LastMask,1);
    agent1_LastState = [LastSparse, LastStd, LastSparseDistribution, Lastsearch_stage]';

    % ----------------------------------------------------------
    % Agent 2: Population size control (State_pop - Last)
    % ----------------------------------------------------------
    agent2_LastState = [LastSparse, LastStd, LastSparseDistribution, Lastsearch_stage]';

    % ----------------------------------------------------------
    % Agent 3: Operator selection (State_op - Last)
    % ----------------------------------------------------------
    Lastobjs = LastPopulation.objs;
    [N, M] = size(Lastobjs);

    LastminObj = min(Lastobjs, [], 1);
    LastmaxObj = max(Lastobjs, [], 1);
    LastrangeObj = LastmaxObj - LastminObj;
    LastrangeObj(LastrangeObj == 0) = 1;

    LastnormObjs = (Lastobjs - repmat(LastminObj, N, 1)) ./ repmat(LastrangeObj, N, 1);

    LastObjMean = mean(LastnormObjs(:));

    LastindSum = sum(LastnormObjs, 2);
    LastObjStd = std(LastindSum);

    % LastObjStd = std(LastindSum ./ (M)) ./ (mean(LastindSum ./ (M)) + 1e-10);

    bins = 0:0.1:1;
    LastObjDistribution = histcounts(LastindSum ./ (M), bins) ./ N;

    agent3_LastState = [LastObjMean, LastObjStd, LastObjDistribution, Lastsearch_stage]';

    % ----------------------------------------------------------
    % Agent 4: Operator parameter control (State_prm - Last)
    % ----------------------------------------------------------
    agent4_LastState = [LastObjMean, LastObjStd, LastObjDistribution, Lastsearch_stage]';
    
    LastState = [agent1_LastState; agent2_LastState; agent3_LastState; agent4_LastState];

    % ==========================================================

    % ==========================================================
    % ----------------------------------------------------------
    % Agent 1: Fitness selection (State_gv - Current)
    % ----------------------------------------------------------
    search_stage = Problem.FE / Problem.maxFE;
    
    CurSparse = mean(Mask(:));
    CurStd    = std(sum(Mask,2))/size(Mask,2);
    CurSparseDistribution = histcounts(sum(Mask,2)./size(Mask,2), 0:0.1:1)./size(Mask,1);
    agent1_CurState = [CurSparse, CurStd, CurSparseDistribution, search_stage]';

    % ----------------------------------------------------------
    % Agent 2: Population size control (State_pop - Current)
    % ----------------------------------------------------------
    agent2_CurState = [CurSparse, CurStd, CurSparseDistribution, search_stage]';

    % ----------------------------------------------------------
    % Agent 3: Operator selection (State_op - Current)
    % ----------------------------------------------------------
    objs = Population.objs;
    [N, M] = size(objs);

    minObj = min(objs, [], 1);
    maxObj = max(objs, [], 1);
    rangeObj = maxObj - minObj;
    rangeObj(rangeObj == 0) = 1;

    normObjs = (objs - repmat(minObj, N, 1)) ./ repmat(rangeObj, N, 1);

    CurObjMean = mean(normObjs(:));

    indSum = sum(normObjs, 2);
    CurObjStd = std(indSum);

    % CurObjStd = std(CurindSum ./ (M)) ./ (mean(CurindSum ./ (M)) + 1e-10);

    bins = 0:0.1:1;
    CurObjDistribution = histcounts(indSum ./ (M), bins) ./ N;

    agent3_CurState = [CurObjMean, CurObjStd, CurObjDistribution, search_stage]';

    % ----------------------------------------------------------
    % Agent 4: Operator parameter control (State_prm - Current)
    % ----------------------------------------------------------
    agent4_CurState = [CurObjMean, CurObjStd, CurObjDistribution, search_stage]';


    CurrentState = [agent1_CurState; agent2_CurState; agent3_CurState; agent4_CurState];
end

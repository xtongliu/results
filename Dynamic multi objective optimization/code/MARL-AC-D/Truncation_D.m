function Del = Truncation_D(PopObj, K)
% Truncation_D
% -------------------------------------------------------------------------
% Truncation operator used by SPEA2 environmental selection.
% -------------------------------------------------------------------------

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

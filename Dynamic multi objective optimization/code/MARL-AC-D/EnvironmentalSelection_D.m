function [Population, Archive, Fitness] = EnvironmentalSelection_D(Population, N)
% EnvironmentalSelection_D
% -------------------------------------------------------------------------
% SPEA2-style environmental selection used as the fine-grained insertion
% mechanism in the SGEA-inspired search module.
% -------------------------------------------------------------------------

    Fitness = CalFitness_D(Population.objs);

    Next    = Fitness < 1;
    Archive = Population(Next);

    if sum(Next) < N
        [~, Rank] = sort(Fitness);
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
        Del  = Truncation_D(Population(Next).objs, sum(Next) - N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
        Archive = Population(Next);
    end

    Population = Population(Next);
    Fitness    = Fitness(Next);
end

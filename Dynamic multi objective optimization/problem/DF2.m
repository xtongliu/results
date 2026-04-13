classdef DF2 < PROBLEM
% <2018> <multi/many> <real> <dynamic>
% taut --- 10 --- Number of generations for static optimization
% nt   --- 10 --- Number of distinct steps
% CEC2018 benchmark dynamic multiobjective optimisation problem DF2.
%
% Reference:
%   S. Jiang, S. Yang, X. Yao, K. C. Tan, M. Kaiser, and N. Krasnogor,
%   "Benchmark Problems for CEC2018 Competition on Dynamic Multiobjective
%   Optimisation", 2018.

    properties
        taut = 10;   % Number of generations for each static optimisation phase (taut)
        nt    = 10;   % Number of distinct time steps (n_t)
    end

    methods
        function Setting(obj)
            [obj.taut,obj.nt] = obj.ParameterSet(10,10);
            obj.M = 2;
            if isempty(obj.D); obj.D = 10; end
            obj.lower    = zeros(1,obj.D);
            obj.upper    = ones(1,obj.D);
            obj.encoding = ones(1,obj.D);
        end
        function Population = Evaluation(obj,Dec)
            PopDec  = obj.CalDec(Dec);
            N       = size(PopDec,1);
            startFE = obj.FE;
            adds    = (startFE + (1:N))';
            PopObj  = obj.CalObj(PopDec,adds);
            PopCon  = obj.CalCon(PopDec);
            Population = SOLUTION(PopDec,PopObj,PopCon,adds);
            if isempty(gcp('nocreate')); obj.FE = startFE + N; end
        end
        function PopObj = CalObj(obj,PopDec,adds)
            t  = floor(adds/obj.N/obj.taut)/obj.nt;
            Gt = abs(sin(0.5*pi*t));
            r  = 1 + floor((obj.D-1)*Gt);
            f1 = zeros(size(PopDec,1),1);
            g  = ones(size(PopDec,1),1);
            for i = 1 : size(PopDec,1)
                ri   = r(i);
                f1(i)= PopDec(i,ri);
                idx  = true(1,obj.D); idx(ri) = false;
                g(i) = 1 + sum((PopDec(i,idx) - Gt(i)).^2);
            end
            f2 = g.*(1 - sqrt(f1./g));
            PopObj = [f1,f2];
        end
        function R = GetOptimum(obj,N)
            R = obj.OptimumAt(N,0);
        end
        function R = GetPF(obj)
            R = obj.GetOptimum(100);
        end
        function score = CalMetric(obj,metName,Population)
            adds  = [Population.add]';
            tau   = floor(adds./obj.N);
            phase = floor(tau./obj.taut);
            change = [0;find(phase(1:end-1)~=phase(2:end));length(phase)];
            Scores = zeros(1,length(change)-1);
            for i = 1 : length(Scores)
                subPop  = Population(change(i)+1:change(i+1));
                t       = phase(change(i)+1)/obj.nt;
                optimum = obj.OptimumAt(1000,t);
                Scores(i) = feval(metName,subPop,optimum);
            end
            score = mean(Scores);
        end
        function DrawObj(obj,Population)
            adds  = [Population.add]';
            tau   = floor(adds./obj.N);
            phase = floor(tau./obj.taut);
            change = [0;find(phase(1:end-1)~=phase(2:end));length(phase)];
            tempStream = RandStream('mlfg6331_64','Seed',2);
            for i = 1 : length(change)-1
                subPop = Population(change(i)+1:change(i+1));
                t      = phase(change(i)+1)/obj.nt;
                color  = rand(tempStream,1,3);
                Draw(subPop.objs+(i-1)*0.1,'o','MarkerSize',5,'MarkerFaceColor',sqrt(color),'MarkerEdgeColor',color);
                Draw(obj.OptimumAt(200,t)+(i-1)*0.1,'-','LineWidth',1,'Color',color);
            end
        end
    end
    methods(Access = private)
        function R = OptimumAt(~,N,~)
            f1 = linspace(0,1,N)';
            f2 = 1 - sqrt(f1);
            R  = [f1,f2];
        end
    end
end

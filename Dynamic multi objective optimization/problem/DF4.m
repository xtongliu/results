classdef DF4 < PROBLEM
% <2018> <multi/many> <real> <dynamic>
% taut --- 10 --- Number of generations for static optimization
% nt   --- 10 --- Number of distinct steps
% CEC2018 benchmark dynamic multiobjective optimisation problem DF4.
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
            obj.lower    = -2*ones(1,obj.D);
            obj.upper    =  2*ones(1,obj.D);
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
            a  = sin(0.5*pi*t);
            b  = 1 + abs(cos(0.5*pi*t));
            Ht = 1.5 + a;
            x1 = PopDec(:,1);
            idx = (2:obj.D);
            target = a.*(x1.^2)./idx;  % implicit expansion
            g  = 1 + sum((PopDec(:,2:end) - target).^2,2);
            f1 = g.*abs(x1 - a).^Ht;
            f2 = g.*abs(x1 - a - b).^Ht;
            PopObj = [f1,f2];
        end
        function R = GetOptimum(obj,N)
            t = floor(obj.FE/obj.N/obj.taut)/obj.nt;
            R = obj.OptimumAt(N,t);
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
        function R = OptimumAt(~,N,t)
            a  = sin(0.5*pi*t);
            b  = 1 + abs(cos(0.5*pi*t));
            Ht = 1.5 + a;
            r  = linspace(0,b,N)';
            f1 = r.^Ht;
            f2 = (b - r).^Ht;
            R  = [f1,f2];
        end
    end
end

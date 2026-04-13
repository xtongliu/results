classdef DF12 < PROBLEM
% <2018> <multi/many> <real> <dynamic>
% taut --- 10 --- Number of generations for static optimization
% nt   --- 10 --- Number of distinct steps
% CEC2018 benchmark dynamic multiobjective optimisation problem DF12.
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
            obj.M = 3;
            if isempty(obj.D); obj.D = 10; end
            obj.D = max(obj.D,3);
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
            kt = 10*sin(pi*t);
            x1 = PopDec(:,1);
            x2 = PopDec(:,2);
            holes = abs( sin(abs(kt.*(2*x1-1))*pi/2) .* sin(abs(kt.*(2*x2-1))*pi/2) );
            core  = sin(t.*x1);
            g = 1 + sum((PopDec(:,3:end) - core).^2,2) + holes;
            f1 = g.*cos(0.5*pi*x1).*cos(0.5*pi*x2);
            f2 = g.*cos(0.5*pi*x1).*sin(0.5*pi*x2);
            f3 = g.*sin(0.5*pi*x1);
            PopObj = [f1,f2,f3];
        end
        function R = GetOptimum(obj,N)
            t = floor(obj.FE/obj.N/obj.taut)/obj.nt;
            R = obj.OptimumAt(N,t);
        end
        function R = GetPF(obj)
            R = obj.GetOptimum(600);
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
            kt = 10*sin(pi*t);
            k  = ceil(sqrt(N));
            u  = linspace(0,1,k);
            [X1,X2] = meshgrid(u,u);
            X1 = X1(:); X2 = X2(:);
            if numel(X1) > N
                X1 = X1(1:N); X2 = X2(1:N);
            end
            holes = abs( sin(abs(kt.*(2*X1-1))*pi/2) .* sin(abs(kt.*(2*X2-1))*pi/2) );
            g = 1 + holes;
            f1 = g.*cos(0.5*pi*X1).*cos(0.5*pi*X2);
            f2 = g.*cos(0.5*pi*X1).*sin(0.5*pi*X2);
            f3 = g.*sin(0.5*pi*X1);
            R  = [f1,f2,f3];
        end
    end
end

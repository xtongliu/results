classdef MAAGENTDMOP < handle
    % MAAGENTDMOP
    % ---------------------------------------------------------------------
    % Multi-agent discrete actor-critic module for dynamic MOO.
    %
    % Redesign highlights:
    %   1) Executed actions are stored as discrete indices.
    %   2) Actors are categorical policies (softmax).
    %   3) Shared critic estimates two state-values from the joint state.
    %   4) Actor updates use policy-gradient style advantages.
    % ---------------------------------------------------------------------

    properties
        numAgents
        numStates
        numActions
        actionType
        actionRange

        allStates

        actors
        critic
        targetCritic

        expPool

        learningRateActor  = 5e-4
        learningRateCritic = 1e-3
        gamma = 0.95
        entropyWeight = 0.01
        targetTau = 0.02

        tActor = 1
        tCritic = 1
        epsilonAdam = 1e-8
        beta1 = 0.9
        beta2 = 0.999

        actorsVelocity
        actorsMomentum
        criticVelocity
        criticMomentum
    end

    methods
        function obj = MAAGENTDMOP(numAgents, numStates, numActions, actionType, actionRange)
            if nargin < 5
                actionRange = {[], []};
            end

            obj.numAgents   = numAgents;
            obj.numStates   = numStates;
            obj.numActions  = numActions;
            obj.actionType  = actionType;
            obj.actionRange = actionRange;
            obj.allStates   = sum(numStates);

            obj.CreateNetworks();
            obj.CreateExpPool();
        end

        function CreateExpPool(obj)
            obj.expPool = struct();
            obj.expPool.size       = 2000;
            obj.expPool.index      = 0;
            obj.expPool.isFull     = false;
            obj.expPool.state      = cell(1, obj.expPool.size);
            obj.expPool.action     = cell(1, obj.expPool.size);      % [numAgents x 1], discrete indices
            obj.expPool.reward     = cell(1, obj.expPool.size);      % [numAgents x 1]
            obj.expPool.next_state = cell(1, obj.expPool.size);
        end

        function Experience(obj, state, actionIdx, reward, next_state)
            obj.expPool.index = obj.expPool.index + 1;
            if obj.expPool.index > obj.expPool.size
                obj.expPool.index = 1;
                obj.expPool.isFull = true;
            end

            idx = obj.expPool.index;
            obj.expPool.state{idx}      = double(state(:));
            obj.expPool.action{idx}     = double(actionIdx(:));
            obj.expPool.reward{idx}     = double(reward(:));
            obj.expPool.next_state{idx} = double(next_state(:));
        end

        function action = Action(obj, state)
            action = [];
            for i = 1:obj.numAgents
                sIdx = obj.getStateIndex(i);
                localState = state(sIdx(1):sIdx(2));
                dlX = dlarray(double(localState(:)), 'CB');
                localAction = extractdata(forward(obj.actors{i}, dlX));
                action = [action; localAction(:)];
            end
        end

        function [responseAction, searchAction] = SelectJointActions(obj, action, Problem)
            acts = obj.SelectAllActions(action, Problem);
            responseAction = acts(1);
            searchAction = acts(2);
        end

        function searchAction = SelectSearchAction(obj, action, Problem)
            searchAction = obj.SelectActionForAgent(action, 2, Problem);
        end

        function acts = SelectAllActions(obj, action, Problem)
            acts = zeros(obj.numAgents, 1);
            for i = 1:obj.numAgents
                acts(i) = obj.SelectActionForAgent(action, i, Problem);
            end
        end

        function act = SelectActionForAgent(obj, action, agentID, Problem)
            exploreProb = obj.getExploreProb(Problem);
            if agentID == 3
                exploreProb = 0.6 * exploreProb;
            elseif agentID == 4
                exploreProb = 0.7 * exploreProb;
            end
            p = obj.getActionSlice(action, agentID);
            if rand < exploreProb
                act = randi(obj.numActions(agentID));
            else
                act = obj.sampleFromProb(p);
            end
        end

        function Train(obj)
            batchSize = 64;

            if obj.expPool.isFull
                totalNum = obj.expPool.size;
            else
                totalNum = obj.expPool.index;
            end

            if totalNum < batchSize
                return;
            end

            indices = randperm(totalNum, batchSize);

            states      = double(cell2mat(obj.expPool.state(indices)));       % [20 x B]
            actionCell  = obj.expPool.action(indices);
            rewards     = double(cell2mat(obj.expPool.reward(indices)));      % [numAgents x B]
            next_states = double(cell2mat(obj.expPool.next_state(indices)));  % [sum(numStates) x B]

            actions = zeros(obj.numAgents, batchSize);
            for i = 1:batchSize
                actions(:, i) = actionCell{i};
            end

            [gradActors, gradCritic] = dlfeval(@obj.lossA2C, ...
                obj.actors, obj.critic, obj.targetCritic, states, actions, rewards, next_states);

            [obj.critic, obj.criticVelocity, obj.criticMomentum] = ...
                obj.updateNetwork(obj.critic, gradCritic, obj.criticVelocity, obj.criticMomentum, obj.learningRateCritic, 'critic');

            for i = 1:obj.numAgents
                [obj.actors{i}, obj.actorsVelocity{i}, obj.actorsMomentum{i}] = ...
                    obj.updateNetwork(obj.actors{i}, gradActors{i}, obj.actorsVelocity{i}, obj.actorsMomentum{i}, obj.learningRateActor, 'actor');
            end

            obj.softUpdateCritic();
        end

        function CreateNetworks(obj)
            obj.actors = cell(1, obj.numAgents);
            obj.actorsVelocity = cell(1, obj.numAgents);
            obj.actorsMomentum = cell(1, obj.numAgents);

            for i = 1:obj.numAgents
                actorLayers = [
                    featureInputLayer(obj.numStates(i), 'Normalization', 'none', 'Name', 'state')
                    fullyConnectedLayer(64, 'Name', 'fc1')
                    reluLayer('Name', 'relu1')
                    fullyConnectedLayer(32, 'Name', 'fc2')
                    reluLayer('Name', 'relu2')
                    fullyConnectedLayer(obj.numActions(i), 'Name', 'logits')
                    softmaxLayer('Name', 'policy')
                ];

                obj.actors{i} = dlnetwork(layerGraph(actorLayers));
                [obj.actorsVelocity{i}, obj.actorsMomentum{i}] = initializeAdamParameters(obj.actors{i}.Learnables.Value);
            end

            criticLayers = [
                featureInputLayer(obj.allStates, 'Normalization', 'none', 'Name', 'joint_state')
                fullyConnectedLayer(128, 'Name', 'fc1')
                reluLayer('Name', 'relu1')
                fullyConnectedLayer(64, 'Name', 'fc2')
                reluLayer('Name', 'relu2')
                fullyConnectedLayer(obj.numAgents, 'Name', 'state_value')
            ];

            obj.critic = dlnetwork(layerGraph(criticLayers));
            obj.targetCritic = obj.critic;
            [obj.criticVelocity, obj.criticMomentum] = initializeAdamParameters(obj.critic.Learnables.Value);
        end

        function [gradThetaActor, gradThetaCritic] = lossA2C(obj, actors, critic, targetCritic, ...
                states, actions, rewards, next_states)

            batchSize = size(states, 2);

            dlStates     = dlarray(states, 'CB');
            dlNextStates = dlarray(next_states, 'CB');
            dlRewards    = dlarray(rewards, 'CB');

            valueNow  = forward(critic, dlStates);           % [2 x B]
            valueNext = forward(targetCritic, dlNextStates); % [2 x B]

            targetValue = dlRewards + obj.gamma * valueNext;
            criticLoss = mean((valueNow - targetValue).^2, 'all');

            % Detach advantages from critic graph
            advData = extractdata(targetValue - valueNow);
            dlAdv   = dlarray(advData, 'CB');

            gradThetaActor = cell(1, obj.numAgents);
            for i = 1:obj.numAgents
                sIdx = obj.getStateIndex(i);
                localState = dlStates(sIdx(1):sIdx(2), :);
                probs = forward(actors{i}, localState);

                chosen = actions(i, :);
                onehotA = zeros(obj.numActions(i), batchSize);
                for b = 1:batchSize
                    onehotA(chosen(b), b) = 1;
                end
                dlOnehot = dlarray(onehotA, 'CB');

                selProb = sum(probs .* dlOnehot, 1);
                entropy = -sum(probs .* log(probs + 1e-8), 1);
                agentAdv = dlAdv(i, :);

                agentWeight = 1.0;
                if i == 3
                    agentWeight = 0.50;
                elseif i == 4
                    agentWeight = 0.60;
                end

                actorLoss = -agentWeight * mean(log(selProb + 1e-8) .* agentAdv + obj.entropyWeight * entropy);
                gradThetaActor{i} = dlgradient(actorLoss, actors{i}.Learnables, 'RetainData', true);
            end

            gradThetaCritic = dlgradient(criticLoss, critic.Learnables);
        end

        function [network, velocities, momenta] = updateNetwork(obj, network, gradients, velocities, momenta, lr, mode)
            if isempty(velocities) || isempty(momenta)
                [velocities, momenta] = initializeAdamParameters(network.Learnables.Value);
            end

            if strcmp(mode, 'critic')
                tNow = obj.tCritic;
            else
                tNow = obj.tActor;
            end

            for i = 1:numel(network.Learnables.Value)
                param = network.Learnables.Value{i};
                grad  = gradients.Value{i};

                if isempty(grad)
                    continue;
                end

                velocity = velocities{i};
                momentum = momenta{i};

                [param, velocity, momentum] = adamupdate( ...
                    param, grad, velocity, momentum, ...
                    tNow, lr, obj.beta1, obj.beta2, obj.epsilonAdam);

                network.Learnables.Value{i} = param;
                velocities{i} = velocity;
                momenta{i} = momentum;
            end

            if strcmp(mode, 'critic')
                obj.tCritic = obj.tCritic + 1;
            else
                obj.tActor = obj.tActor + 1;
            end
        end

        function softUpdateCritic(obj)
            for i = 1:numel(obj.critic.Learnables.Value)
                src = obj.critic.Learnables.Value{i};
                tgt = obj.targetCritic.Learnables.Value{i};
                obj.targetCritic.Learnables.Value{i} = obj.targetTau * src + (1 - obj.targetTau) * tgt;
            end
        end

        function idx = getStateIndex(obj, agentID)
            if agentID == 1
                s1 = 1;
            else
                s1 = sum(obj.numStates(1:agentID-1)) + 1;
            end
            s2 = sum(obj.numStates(1:agentID));
            idx = [s1, s2];
        end

        function p = getActionSlice(obj, action, agentID)
            startIdx = sum(obj.numActions(1:agentID-1)) + 1;
            endIdx   = sum(obj.numActions(1:agentID));
            p = action(startIdx:endIdx);
        end

        function p = getExploreProb(obj, Problem)
            ratio = Problem.FE / max(Problem.maxFE, 1);
            p = max(0.05, 0.35 * (1 - ratio));
        end

        function a = sampleFromProb(obj, pVec)
            p = double(pVec(:));
            p(p < 0) = 0;
            if sum(p) <= 0
                p = ones(size(p)) / numel(p);
            else
                p = p / sum(p);
            end
            c = cumsum(p);
            r = rand;
            a = find(r <= c, 1, 'first');
            if isempty(a)
                a = numel(p);
            end
        end
    end
end

function [velocity, momentum] = initializeAdamParameters(parameters)
    velocity = cell(size(parameters));
    momentum = cell(size(parameters));
    for i = 1:numel(parameters)
        velocity{i} = zeros(size(parameters{i}), 'like', parameters{i});
        momentum{i} = zeros(size(parameters{i}), 'like', parameters{i});
    end
end

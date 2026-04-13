classdef MAAGENT < handle


    properties
        numAgents
        numStates
        numActions
        actionType
        actionRange
        allStates;
        allActions;
        actors
        critic
        expPool
        learningRate
        t=1;
        epsilon = 1e-8
        actorsVelocity;
        actorsMomentum;
        criticVelocity;
        criticMomentum;
        beta1 = 0.9;
        beta2 = 0.999;
    end

    methods
        function obj = MAAGENT(numAgents, numStates, numActions, actionType, actionRange)

            obj.numAgents = numAgents;
            obj.numStates = numStates;
            obj.numActions = numActions;
            obj.actionType = actionType;
            obj.actionRange = actionRange;
            obj.allStates = sum(numStates);
            obj.allActions = sum(numActions);
            obj.learningRate = 1e-3;
            obj.t = 1;

            obj.CreateNetworks();
            obj.CreateExpPool();
        end

        function CreateExpPool(obj)
            obj.expPool = struct();
            obj.expPool.size = 1000;
            obj.expPool.index = 0;
            obj.expPool.isFull = false;
            obj.expPool.state      = cell(1, obj.expPool.size);
            obj.expPool.action     = cell(1, obj.expPool.size);
            obj.expPool.reward     = cell(1, obj.expPool.size);
            obj.expPool.next_state = cell(1, obj.expPool.size);
            obj.expPool.numAllS    = 0;
        end

        function Experience(obj, state, action, reward, next_state)

            obj.expPool.index = obj.expPool.index + 1;
            if obj.expPool.index > obj.expPool.size
                obj.expPool.index = 1;
                obj.expPool.isFull = true;
            end
            idx = obj.expPool.index;
            
            obj.expPool.state{idx}      = double(state);
            obj.expPool.action{idx}     = double(action);
            obj.expPool.reward{idx}     = double(reward);
            obj.expPool.next_state{idx} = double(next_state);
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


            states      = double(cell2mat(obj.expPool.state(indices)));
            actions = double(cell2mat(obj.expPool.action(indices)));

            rewards = double(cell2mat(obj.expPool.reward(indices)));
            next_states = double(cell2mat(obj.expPool.next_state(indices)));


            [grad_actors, grad_critic] = dlfeval(@obj.lossCritic, obj.actors, obj.critic, states, actions, rewards, next_states);


            [obj.critic, obj.criticVelocity, obj.criticMomentum] = obj.updateNetwork(obj.critic, grad_critic, obj.criticVelocity, obj.criticMomentum);
            for j = 1:obj.numAgents
               [obj.actors{j}, obj.actorsVelocity{j}, obj.actorsMomentum{j}] = obj.updateNetwork(obj.actors{j}, grad_actors{j}, obj.actorsVelocity{j}, obj.actorsMomentum{j});
            end
        end

        function action = Action(obj, state)


            action = [];
            for i = 1:obj.numAgents
                if i == 1
                    start_idx = 1;
                else
                    start_idx = sum(obj.numStates(1:i-1)) + 1;
                end
                end_idx = sum(obj.numStates(1:i));
                agent_state = state(start_idx:end_idx);
                dlX = dlarray(agent_state, 'CB');
                agent_action = extractdata(forward(obj.actors{i}, dlX));
                if obj.actionType(i) == 0 && ~isempty(obj.actionRange{i})
                    range = obj.actionRange{i};
                    agent_action = (agent_action + 1) / 2 * (range(2) - range(1)) + range(1);
                end
                action = [action; agent_action];
            end
        end

        function [fitnessaction, selectratio, operatoraction, paramaction] = SelectActions(obj, action, Problem)

            explore_prob = 0.3^(Problem.FE/Problem.maxFE);
            min_exp = ceil(0.25 * Problem.maxFE / 100);
            exp_ready = obj.expPool.isFull || obj.expPool.index > min_exp;

            agent1_action = action(1:3);
            agent2_action = action(4);
            agent3_action = action(5:7);
            agent4_action = action(8);

            if rand < explore_prob || ~exp_ready

                fitnessaction = randsample(1:3, 1, true);
                selectratio = 0.5 + 0.5 * rand();
                operatoraction = randsample(1:3, 1, true);
                paramaction = -1 + 2*rand();
            else

                [~, fitnessaction] = max(agent1_action);
                selectratio = agent2_action;
                [~, operatoraction] = max(agent3_action);
                paramaction = agent4_action;
            end
        end
 
        function CreateNetworks(obj)

            obj.actors = cell(1, obj.numAgents);
            obj.actorsVelocity = cell(1, obj.numAgents);
            obj.actorsMomentum = cell(1, obj.numAgents);
            

            actor_layers = cell(1, obj.numAgents); 
            for i = 1:obj.numAgents
                if obj.actionType(i) == 1
                    actor_layers{i} = [
                        featureInputLayer(obj.numStates(i), 'Normalization', 'none', 'Name', 'state')
                        fullyConnectedLayer(64, 'Name', 'fc1')
                        reluLayer('Name', 'relu1')
                        fullyConnectedLayer(32, 'Name', 'fc2')
                        reluLayer('Name', 'relu2')
                        fullyConnectedLayer(obj.numActions(i), 'Name', 'logits')
                        softmaxLayer('Name', 'action_probs')
                    ];
                else
                    actor_layers{i} = [
                        featureInputLayer(obj.numStates(i), 'Normalization', 'none', 'Name', 'state')
                        fullyConnectedLayer(64, 'Name', 'fc1')
                        reluLayer('Name', 'relu1')
                        fullyConnectedLayer(32, 'Name', 'fc2')
                        reluLayer('Name', 'relu2')
                        fullyConnectedLayer(obj.numActions(i), 'Name', 'mu')
                        tanhLayer('Name', 'mu_out')
                    ];
                end
                obj.actors{i} = dlnetwork(layerGraph(actor_layers{i}));
                obj.actorsVelocity{i} = initializeAdamParameters(obj.actors{i}.Learnables.Value);
                obj.actorsMomentum{i} = initializeAdamParameters(obj.actors{i}.Learnables.Value);
            end


            critic_layers = [
                featureInputLayer(obj.allStates + obj.allActions, 'Normalization', 'none', 'Name', 'state')
                fullyConnectedLayer(256, 'Name', 'fc1')
                reluLayer('Name', 'relu1')
                fullyConnectedLayer(128, 'Name', 'fc2')
                reluLayer('Name', 'relu2')
                fullyConnectedLayer(1, 'Name', 'value')
            ];
            obj.critic = dlnetwork(layerGraph(critic_layers));

            obj.criticVelocity = initializeAdamParameters(obj.critic.Learnables.Value);
            obj.criticMomentum = initializeAdamParameters(obj.critic.Learnables.Value);
        end

        function [grad_theta_actor, grad_theta_critic] = lossCritic(obj, actors, critic, states, actions, rewards, next_states)           
            statesSepar = cell(1, obj.numAgents);
            actionsSepar = cell(1, obj.numAgents);
            next_statesSepar  = cell(1, obj.numAgents);
        
            dlStates = dlarray(states, 'CB');
            dlActions = dlarray(actions, 'CB');
            dlRewards = dlarray(rewards, 'CB');
            dlNext_states = dlarray(next_states, 'CB');
            
            %action states
            actionStates = [];
            actionNext_states = [];
         
            for i = 1:obj.numAgents
                statesSepar{i} = states(sum(obj.numStates(1:i-1))+1:sum(obj.numStates(1:i)),:);
                actionsSepar{i} = actions(sum(obj.numActions(1:i-1))+1:sum(obj.numActions(1:i)),:);
                next_statesSepar{i} = next_states(sum(obj.numStates(1:i-1))+1:sum(obj.numStates(1:i)),:);
                actionStates = [actionStates;forward(actors{i}, dlarray(statesSepar{i}, 'CB'))];
                actionNext_states = [actionNext_states;forward(actors{i}, dlarray(next_statesSepar{i}, 'CB'))];
            end
                

            actor_loss = mean(-forward(critic,[dlStates;actionStates]));
            grad_theta_actor = cell(1, obj.numAgents);
            for i = 1:obj.numAgents                                              
                grad_theta_actor{i} = dlgradient(actor_loss, actors{i}.Learnables);
            end
                                          

            critic_loss = mean((forward(critic,[dlStates; dlActions]) - (forward(critic,[dlNext_states; actionNext_states]) + dlRewards)).^2);
            grad_theta_critic = dlgradient(critic_loss, critic.Learnables);    
        end

        function [network, velocities, momenta] = updateNetwork(obj, network, gradients, velocities, momenta)

            if isempty(velocities) || isempty(momenta)
                velocities = initializeAdamParameters(network.Learnables.Value);
                momenta = initializeAdamParameters(network.Learnables.Value);
            end

            for i = 1:numel(network.Learnables.Value)

                param = network.Learnables.Value{i};
                grad = gradients.Value{i};
                velocity = velocities{i};
                momentum = momenta{i};

                [param, velocity, momentum] = adamupdate(param, grad, velocity, momentum, obj.t, obj.learningRate, obj.beta1, obj.beta2, obj.epsilon);

                network.Learnables.Value{i} = param;
                velocities{i} = velocity;
                momenta{i} = momentum;
            end

            obj.t = obj.t + 1;
        end
    end
end

function [velocity, momentum] = initializeAdamParameters(parameters)
    velocity = cell(size(parameters));
    momentum = cell(size(parameters));
    for i = 1:numel(parameters)
        velocity{i} = zeros(size(parameters{i}));
        momentum{i} = zeros(size(parameters{i}));
    end
end

--[[
    Load VGG16 + LSTM network.
]]


require 'nn'

------------------------------------------------------------------------------------------------------------

local function SelectFeatsDisableBackprop(net)
    local features = net
    features:remove(features:size()) -- remove logsoftmax layer
    features:remove(features:size()) -- remove 3rd linear layer
    return nn.NoBackprop(features)
end

------------------------------------------------------------------------------------------------------------

local function load_features_network()
    local filepath = paths.concat(projectDir, 'data', 'pretrained_models')

    local net = torch.load(paths.concat(filepath, 'model_vgg16.t7'))
    local params = torch.load(paths.concat(filepath, 'parameters_vgg16.t7'))

    net:remove(net:size()) -- remove logsoftmax layer
    net:remove(net:size()) -- remove 3rd linear layer

    params.feat_size = 4096

    return net, params
end

------------------------------------------------------------------------------------------------------------

local function load_classifier_network(input_size, num_feats, num_activities, num_layers, batchSize, seq_length)
    local view = nn.View(batchSize * seq_length, -1)
    local lstm = nn.Sequential()
    lstm:add(nn.Contiguous())
    lstm:add(cudnn.LSTM(input_size, num_feats, num_layers, true))
    lstm:add(nn.Contiguous())
    lstm:add(view)
    lstm:add(nn.Linear(num_feats, num_activities))
    return lstm, view
end

------------------------------------------------------------------------------------------------------------

--[[ Create VGG16 + LSTM ]]--
local function create_network()

    local vgg16, params = load_features_network()
    vgg16:evaluate()

    local lstm, view3 = load_classifier_network(params.feat_size,
                                                opt.nFeats,
                                                opt.num_activities,
                                                opt.nLayers,
                                                opt.batchSize,
                                                opt.seq_length)

    local model = nn.Sequential()
    model:add(nn.View(opt.batchSize * opt.seq_length, 3, 224, 224))
    model:add(SelectFeatsDisableBackprop(vgg16))
    model:add(nn.View(opt.batchSize, opt.seq_length, -1))
    model:add(lstm)
    model.view3 = view3

    -- monkey-patch the training function
    function model:training()
        model.modules[4]:training()
    end

    return model, params
end

------------------------------------------------------------------------------------------------------------

return create_network
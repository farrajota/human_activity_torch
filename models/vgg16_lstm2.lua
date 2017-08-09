--[[
    Load VGG16 + LSTM network.
]]


require 'nn'


------------------------------------------------------------------------------------------------------------

local function load_features_network()
    local filepath = paths.concat(projectDir, 'data', 'pretrained_models')

    local net = torch.load(paths.concat(filepath, 'model_vgg16.t7'))
    local params = torch.load(paths.concat(filepath, 'parameters_vgg16.t7'))

    net:remove(net:size()) -- remove logsoftmax layer
    net:remove(net:size()) -- remove 3rd linear layer
    net:remove(net:size()) -- remove 2nd dropout layer
    net:remove(net:size()) -- remove 2nd last relu layer
    net:remove(net:size()) -- remove 2nd linear layer
    net:remove(net:size()) -- remove 1st dropout layer
    net:remove(net:size()) -- remove 1st relu layer
    net:remove(net:size()) -- remove 1st linear layer
    net:remove(net:size()) -- remove view layer

    params.feat_size = 512

    return net, params
end

------------------------------------------------------------------------------------------------------------

local function load_classifier_network(input_size, num_feats, num_activities, num_layers)
    local lstm = nn.Sequential()
    lstm:add(nn.VolumetricAveragePooling(1,7,7,1,1,1))
    lstm:add(nn.Squeeze())
    lstm:add(nn.Contiguous())
    lstm:add(cudnn.LSTM(input_size, num_feats, num_layers, true))
    lstm:add(nn.Contiguous())
    lstm:add(nn.View(-1, num_feats))
    lstm:add(nn.Linear(num_feats, num_activities))
    return lstm
end

------------------------------------------------------------------------------------------------------------

--[[ Create VGG16 + LSTM ]]--
local function create_network()

    local features, params = load_features_network()
    features:evaluate()

    local lstm = load_classifier_network(params.feat_size,
                                         opt.nFeats,
                                         opt.num_activities,
                                         opt.nLayers)

    return features, nil, lstm, params  -- features, kps, classifier, params
end

------------------------------------------------------------------------------------------------------------

return create_network
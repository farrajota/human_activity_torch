--[[
    Load VGG16 + ConvNet (avg pool + lin layer) networks.
]]



require 'nn'


------------------------------------------------------------------------------------------------------------

local function load_features_network()
    local filepath = paths.concat(projectDir, 'data', 'pretrained_models')

    local net = torch.load(paths.concat(filepath, 'model_vgg16.t7'))
    local params = torch.load(paths.concat(filepath, 'parameters_vgg16.t7'))

    net:remove(net:size())  -- remove logsoftmax layer
    net:remove(net:size())  -- remove 3rd linear layer

    params.feat_size = 4096

    return net, params
end

------------------------------------------------------------------------------------------------------------

local function load_classifier_network(input_size, num_activities, seq_length)
    local classifier = nn.Sequential()
    classifier:add(nn.View(seq_length, 1, input_size))  -- convert to a nx1xfeats format
    classifier:add(nn.SpatialAveragePooling(seq_length, 1, 1, 1))
    classifier:add(nn.View(-1, input_size))  -- convert to batchsize x feats format
    classifier:add(nn.Linear(input_size, num_activities))
    return classifier
end

------------------------------------------------------------------------------------------------------------

--[[ Create VGG16 + spatial average pooling + lin layer ]]--
local function create_network()

    local features, params = load_features_network()
    features:evaluate()

    local classifier = load_classifier_network(params.feat_size,
                                               opt.num_activities,
                                               opt.seq_length)

    return features, nil, classifier, params  -- features, kps, classifier, params
end

------------------------------------------------------------------------------------------------------------

return create_network
--[[
    Load VGG16 + ConvNet (avg pool + lin layer) networks.
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

local function load_classifier_network(input_size, num_feats, num_activities, num_layers, seq_length)
    local classifier = nn.Sequential()
    --classifier:add(nn.Unsqueeze(2))   -- add singleton to dim2 such that input = B x 1 x seq_length x input_size
    classifier:add(nn.Transpose({2,3}))  -- swap dim2 with dim4 such that input = B x input_size x seq_length x 1
    classifier:add(nn.VolumetricConvolution(input_size, num_feats, 5,5,5, 1,1,1, 2,2,2))
    classifier:add(nn.ReLU(true))
    --classifier:add(nn.VolumetricMaxPooling(2,2,2, 2,2,2))
    for i=2, num_layers do
        classifier:add(nn.VolumetricConvolution(num_feats, num_feats, 5,5,5, 1,1,1, 2,2,2))
        classifier:add(nn.ReLU(true))
        --classifier:add(nn.VolumetricMaxPooling(2,2,2, 2,2,2))
    end
    classifier:add(nn.VolumetricAveragePooling(seq_length,7,7, 1,1,1))
    classifier:add(nn.VolumetricConvolution(num_feats, num_activities, 1,1,1, 1,1,1))
    classifier:add(nn.View(-1, num_activities))

    --classifier:add(nn.View(seq_length, 1, input_size))  -- convert to a BxNx1xfeats format
    --classifier:add(nn.SpatialAveragePooling(seq_length, 1, 1, 1))
    --classifier:add(nn.View(-1, input_size))  -- convert to batchsize x feats format
    --classifier:add(nn.TemporalConvolution(input_size, num_activities, 1, 1))
    --classifier:add(nn.View(-1, input_size))  -- convert to batchsize x feats format
    --classifier:add(nn.Linear(input_size, num_activities))
    return classifier
end

------------------------------------------------------------------------------------------------------------

--[[ Create VGG16 + spatial average pooling + lin layer ]]--
local function create_network()

    local features, params = load_features_network()
    features:evaluate()

    local classifier = load_classifier_network(params.feat_size,
                                               opt.nFeats,
                                               opt.num_activities,
                                               opt.nLayers,
                                               opt.seq_length)

    return features, nil, classifier, params  -- features, kps, classifier, params
end

------------------------------------------------------------------------------------------------------------

return create_network
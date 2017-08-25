--[[
    Load hms + ConvNet (avg pool + lin layer) networks.
]]



require 'nn'


------------------------------------------------------------------------------------------------------------

local function load_classifier_network(input_size, num_feats, num_activities, num_layers, seq_length)
    local classifier = nn.Sequential()
    classifier:add(nn.Transpose({2,3}))  -- swap dim2 with dim4 such that input = B x input_size x seq_length x 1
    classifier:add(nn.VolumetricConvolution(input_size, num_feats, 3,3,3, 1,1,1, 1,1,1))
    classifier:add(nn.VolumetricBatchNormalization(num_feats, 1e-3))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.VolumetricMaxPooling(2,2,2, 2,2,2))
    for i=2, num_layers do
        classifier:add(nn.VolumetricConvolution(num_feats, num_feats, 3,3,3, 1,1,1, 1,1,1))
        classifier:add(nn.VolumetricBatchNormalization(num_feats, 1e-3))
        classifier:add(nn.ReLU(true))
        classifier:add(nn.VolumetricMaxPooling(2,2,2, 2,2,2))
    end
    --classifier:add(nn.VolumetricAveragePooling(seq_length, 64,64, 1,1,1))
    classifier:add(nn.VolumetricAveragePooling(math.floor(seq_length/(2*num_layers)), 64/(2*num_layers),64/(2*num_layers), 1,1,1))
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

    local hms_features, params = paths.dofile('../load_hg_best.lua')()
    hms_features:evaluate()

    local classifier = load_classifier_network(params.feat_size,
                                               opt.nFeats,
                                               opt.num_activities,
                                               opt.nLayers,
                                               opt.seq_length)

    return nil, hms_features, classifier, params  -- features, hms, classifier, params
end

------------------------------------------------------------------------------------------------------------

return create_network
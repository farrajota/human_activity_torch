--[[
    Load hms + ConvNet (5d conv + lin layer) networks.
]]



require 'nn'


------------------------------------------------------------------------------------------------------------

local function load_classifier_network(input_size, num_feats, num_activities, num_layers, seq_length)
    local classifier = nn.Sequential()
    classifier:add(nn.Transpose({2,3}))  -- swap dim2 with dim4 such that input = B x input_size x seq_length x 1
    classifier:add(nn.VolumetricConvolution(input_size, 128, 3,3,3, 1,1,1, 1,1,1))  -- 64 -> 64
    classifier:add(nn.VolumetricBatchNormalization(128, 1e-3))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.VolumetricMaxPooling(2,2,2, 2,2,2))  -- 64 -> 32
    classifier:add(nn.VolumetricConvolution(128, num_feats, 3,3,3, 1,1,1, 1,1,1))
    classifier:add(nn.VolumetricBatchNormalization(num_feats, 1e-3))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.VolumetricMaxPooling(2,2,2, 2,2,2))  -- 32 -> 16
    classifier:add(nn.VolumetricConvolution(num_feats, num_feats+128, 3,3,3, 1,1,1, 1,1,1))
    classifier:add(nn.VolumetricBatchNormalization(num_feats+128, 1e-3))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.VolumetricMaxPooling(2,2,2, 2,2,2))  -- 16 -> 8
    classifier:add(nn.VolumetricConvolution(num_feats+128, 512, 3,3,3, 1,1,1, 1,1,1))
    classifier:add(nn.VolumetricBatchNormalization(512, 1e-3))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.VolumetricMaxPooling(2,2,2, 2,2,2))  -- 8 -> 4
    classifier:add(nn.VolumetricConvolution(512, num_activities, 2,4,4, 1,1,1, 0,0,0))
    classifier:add(nn.View(-1, num_activities))
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
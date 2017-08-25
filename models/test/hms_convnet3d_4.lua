--[[
    Load hms + ConvNet (avg pool + lin layer) networks.
]]



require 'nn'


------------------------------------------------------------------------------------------------------------

local function load_classifier_network(input_size, num_feats, num_activities, num_layers, seq_length)
    local classifier = nn.Sequential()
    classifier:add(nn.Transpose({2,3}))  -- swap dim2 with dim4 such that input = B x input_size x seq_length x 1
    classifier:add(nn.VolumetricConvolution(input_size, num_feats, 7,7,7, 2,2,2, 3,3,3))
    classifier:add(nn.VolumetricBatchNormalization(num_feats, 1e-3))
    classifier:add(nn.ReLU(true))
    --classifier:add(nn.VolumetricMaxPooling(2,2,2, 2,2,2))
    classifier:add(nn.Dropout(opt.dropout))
    for i=2, num_layers do
        classifier:add(nn.VolumetricConvolution(num_feats, num_feats, 7,7,7, 2,2,2, 3,3,3))
        classifier:add(nn.VolumetricBatchNormalization(num_feats, 1e-3))
        classifier:add(nn.ReLU(true))
        --classifier:add(nn.VolumetricMaxPooling(2,2,2, 2,2,2))
        classifier:add(nn.Dropout(opt.dropout))
    end
    classifier:add(nn.VolumetricConvolution(num_feats, num_activities,
                                            math.ceil(seq_length/(2*num_layers)),64/(2*num_layers),64/(2*num_layers),
                                            1,1,1))
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
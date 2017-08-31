--[[
    Load Resnet-50 + LSTM network.
]]


require 'nn'


------------------------------------------------------------------------------------------------------------

local function load_classifier_network(input_size, num_feats, num_activities, num_layers)
    local lstm = nn.Sequential()
    lstm:add(nn.Contiguous())
    lstm:add(cudnn.LSTM(input_size, num_feats, num_layers, true))
    lstm:add(nn.Contiguous())
    lstm:add(nn.View(-1, num_feats))
    lstm:add(nn.Dropout(opt.dropout))
    lstm:add(nn.Linear(num_feats, num_activities))
    return lstm
end

------------------------------------------------------------------------------------------------------------

--[[ Create VGG16 + LSTM ]]--
local function create_network()

    local features, params = paths.dofile('../load_resnet.lua')('resnet50')
    features:evaluate()

    local lstm = load_classifier_network(params.feat_size,
                                         opt.nFeats,
                                         opt.num_activities,
                                         opt.nLayers)

    return features, nil, lstm, params  -- features, hms, classifier, params
end

------------------------------------------------------------------------------------------------------------

return create_network
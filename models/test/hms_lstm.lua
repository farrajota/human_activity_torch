--[[
    Load Body joints predictor + LSTM networks.
]]

--[[
    Load VGG16 + LSTM network.
]]


require 'nn'


------------------------------------------------------------------------------------------------------------

local function load_classifier_network(input_size, num_feats, num_activities, num_layers)
    local lstm = nn.Sequential()
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

    local hms_features, params = paths.dofile('../load_hg_best.lua')()
    hms_features:evaluate()

    local lstm = load_classifier_network(params.feat_size,
                                         opt.nFeats,
                                         opt.num_activities,
                                         opt.nLayers)

    return nil, hms_features, lstm, params  -- features, hms, classifier, params
end

------------------------------------------------------------------------------------------------------------

return create_network
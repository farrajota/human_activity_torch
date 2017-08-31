--[[
    Load VGG16 + Body joints predictor + LSTM networks.

    Classifier type: single lstm.
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

    local vgg16 = paths.dofile('vgg16_lstm.lua')
    local hms = paths.dofile('hms_lstm.lua')

    local vgg16_features, _, vgg16_lstm, vgg16_params = vgg16()
    local _, hms_features, hms_lstm, hms_params = hms()
    local hms_feat_size = hms_params.dims[1] * hms_params.dims[2] * hms_params.dims[3]

    local lstm = load_classifier_network(vgg16_params.feat_size + hms_feat_size,
                                         opt.nFeats,
                                         opt.num_activities,
                                         opt.nLayers)


    local classifier = nn.Sequential()
    classifier:add(nn.JoinTable(3))
    classifier:add(lstm)

    return vgg16_features, hms_features, classifier, vgg16_params  -- features, hms, classifier, params
end

------------------------------------------------------------------------------------------------------------

return create_network
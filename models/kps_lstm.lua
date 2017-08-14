--[[
    Load Body joints predictor + LSTM networks.
]]

--[[
    Load VGG16 + LSTM network.
]]


require 'nn'


------------------------------------------------------------------------------------------------------------

local function load_features_network()
    local filepath = paths.concat(projectDir, 'data', 'pretrained_models')
    local hg_net = torch.load(paths.concat(filepath, 'hg-best.t7'))
    local net = nn.Sequential()
    net:add(hg_net)
    net:add(nn.SelectTable(-1))
    net:add(nn.View(-1, 11*64*64))
    local params = {
      pixel_scale = 1,
      dims = {11, 64, 64},
      feat_size = 11 * 64 * 64
    }
    return net, params
end

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

    local kps_features, params = load_features_network()
    kps_features:evaluate()

    local lstm = load_classifier_network(params.feat_size,
                                         opt.nFeats,
                                         opt.num_activities,
                                         opt.nLayers)

    return nil, kps_features, lstm, params  -- features, kps, classifier, params
end

------------------------------------------------------------------------------------------------------------

return create_network
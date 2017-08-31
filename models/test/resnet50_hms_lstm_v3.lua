--[[
    Load resnet50 + Body joints predictor + LSTM networks.

    Classifier type: add/merge two lstm+linear layer results.
]]


------------------------------------------------------------------------------------------------------------

--[[ Create resnet50 + LSTM ]]--
local function create_network()

    local resnet50 = paths.dofile('resnet50_lstm.lua')
    local hms = paths.dofile('hms_lstm.lua')

    local resnet50_features, _, resnet50_lstm, resnet50_params = resnet50()
    local _, hms_features, hms_lstm, _ = hms()

    local classifier = nn.Sequential()
    classifier:add(nn.ConcatTable()
        :add(nn.Sequential():add(nn.SelectTable(1)):add(resnet50_lstm))
        :add(nn.Sequential():add(nn.SelectTable(2)):add(hms_lstm)))
    classifier:add(nn.CAddTable())

    return resnet50_features, hms_features, classifier, resnet50_params  -- features, hms, classifier, params
end

------------------------------------------------------------------------------------------------------------

return create_network
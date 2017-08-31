--[[
    Load VGG16 + Body joints predictor + LSTM networks.

    Classifier type: add/merge two lstm+linear layer results.
]]


------------------------------------------------------------------------------------------------------------

--[[ Create VGG16 + LSTM ]]--
local function create_network()

    local vgg16 = paths.dofile('vgg16_lstm.lua')
    local hms = paths.dofile('hms_lstm.lua')

    local vgg16_features, _, vgg16_lstm, vgg16_params = vgg16()
    local _, hms_features, hms_lstm, _ = hms()

    local classifier = nn.Sequential()
    classifier:add(nn.ConcatTable()
        :add(nn.Sequential():add(nn.SelectTable(1)):add(vgg16_lstm))
        :add(nn.Sequential():add(nn.SelectTable(2)):add(hms_lstm)))
    classifier:add(nn.CAddTable())

    return vgg16_features, hms_features, classifier, vgg16_params  -- features, hms, classifier, params
end

------------------------------------------------------------------------------------------------------------

return create_network
--[[
    Load VGG16 + Body joints predictor + LSTM networks.

    Classifier type: merge two lstm results into a single Linear layer.
]]


require 'nn'



------------------------------------------------------------------------------------------------------------

--[[ Create VGG16 + LSTM ]]--
local function create_network()

    local vgg16 = paths.dofile('vgg16_lstm.lua')
    local hms = paths.dofile('hms_lstm.lua')

    local vgg16_features, _, vgg16_lstm, vgg16_params = vgg16()
    local _, hms_features, hms_lstm, hms_params = hms()

    -- remove the linear layer
    vgg16_lstm:remove(vgg16_lstm:size())
    hms_lstm:remove(hms_lstm:size())

    local classifier = nn.Sequential()
    classifier:add(nn.ConcatTable()
        :add(nn.Sequential():add(nn.SelectTable(1)):add(vgg16_lstm))
        :add(nn.Sequential():add(nn.SelectTable(2)):add(hms_lstm)))
    classifier:add(nn.CAddTable())
    classifier:add(nn.Linear(opt.nFeats, opt.num_activities))

    return vgg16_features, hms_features, classifier, vgg16_params  -- features, hms, classifier, params
end

------------------------------------------------------------------------------------------------------------

return create_network
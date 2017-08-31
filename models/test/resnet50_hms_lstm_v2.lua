--[[
    Load resnet50 + Body joints predictor + LSTM networks.

    Classifier type: merge two lstm results into a single Linear layer.
]]


require 'nn'



------------------------------------------------------------------------------------------------------------

--[[ Create resnet50 + LSTM ]]--
local function create_network()

    local resnet50 = paths.dofile('resnet50_lstm.lua')
    local hms = paths.dofile('hms_lstm.lua')

    local resnet50_features, _, resnet50_lstm, resnet50_params = resnet50()
    local _, hms_features, hms_lstm, hms_params = hms()

    resnet50_lstm:remove(resnet50_lstm:size())
    hms_lstm:remove(hms_lstm:size())

    local classifier = nn.Sequential()
    classifier:add(nn.ConcatTable()
        :add(nn.Sequential():add(nn.SelectTable(1)):add(resnet50_lstm))
        :add(nn.Sequential():add(nn.SelectTable(2)):add(hms_lstm)))
    classifier:add(nn.CAddTable())
    classifier:add(nn.Linear(opt.nFeats, opt.num_activities))

    return resnet50_features, hms_features, classifier, resnet50_params  -- features, hms, classifier, params
end

------------------------------------------------------------------------------------------------------------

return create_network
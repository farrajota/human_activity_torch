--[[
    Load VGG16 + Body joints predictor + Temporal ConvNet networks.
]]


------------------------------------------------------------------------------------------------------------

--[[ Create VGG16 + LSTM ]]--
local function create_network()

    local vgg16 = paths.dofile('vgg16_convnet3d.lua')
    local hms = paths.dofile('hms_convnet3d.lua')

    local vgg16_features, _, vgg16_convnet, vgg16_params = vgg16()
    local _, hms_features, hms_convnet, _ = hms()

    local classifier = nn.Sequential()
    classifier:add(nn.ConcatTable()
        :add(nn.Sequential():add(nn.SelectTable(1)):add(vgg16_convnet))
        :add(nn.Sequential():add(nn.SelectTable(2)):add(hms_convnet)))
    classifier:add(nn.CAddTable())

    return vgg16_features, hms_features, classifier, vgg16_params  -- features, hms, classifier, params
end

------------------------------------------------------------------------------------------------------------

return create_network
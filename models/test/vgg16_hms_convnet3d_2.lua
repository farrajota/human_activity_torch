--[[
    Load VGG16 + Body joints predictor + Temporal ConvNet networks.
]]


------------------------------------------------------------------------------------------------------------

--[[ Create VGG16 + LSTM ]]--
local function create_network()

    local vgg16 = paths.dofile('vgg16_convnet3d_2.lua')
    local kps = paths.dofile('kps_convnet3d_2.lua')

    local vgg16_features, _, vgg16_convnet, vgg16_params = vgg16()
    local _, kps_features, kps_convnet, _ = kps()

    local classifier = nn.Sequential()
    classifier:add(nn.ConcatTable()
        :add(nn.Sequential():add(nn.SelectTable(1)):add(vgg16_convnet))
        :add(nn.Sequential():add(nn.SelectTable(2)):add(kps_convnet)))
    classifier:add(nn.CAddTable())

    return vgg16_features, kps_features, classifier, vgg16_params  -- features, kps, classifier, params
end

------------------------------------------------------------------------------------------------------------

return create_network
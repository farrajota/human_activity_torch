--[[
    Load hms + ConvNet (5d conv + lin layer) networks.
]]



require 'nn'


------------------------------------------------------------------------------------------------------------

local conv = nn.VolumetricConvolution
local batchnorm = nn.VolumetricBatchNormalization
local relu = nn.ReLU

-- Main convolutional block
local function convBlock(numIn,numOut)
    return nn.Sequential()
        :add(conv(numIn,numOut/2, 1,1,1, 1,1,1, 0,0,0))
        :add(batchnorm(numOut/2))
        :add(relu(true))
        :add(conv(numOut/2,numOut/2, 3,3,3, 1,1,1, 1,1,1))
        :add(batchnorm(numOut/2))
        :add(relu(true))
        :add(conv(numOut/2,numOut, 1,1,1, 1,1,1, 0,0,0))
        :add(batchnorm(numOut))
end

-- Skip layer
local function skipLayer(numIn,numOut)
    if numIn == numOut then
        return nn.Identity()
    else
        return nn.Sequential()
            :add(conv(numIn,numOut, 1,1,1, 1,1,1, 0,0,0))
            :add(batchnorm(numOut))
    end
end

-- Residual block
function Residual(numIn,numOut)
    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(convBlock(numIn,numOut))
            :add(skipLayer(numIn,numOut)))
        :add(nn.CAddTable(true))
        :add(relu(true))
end

------------------------------------------------------------------------------------------------------------

local function load_classifier_network(input_size, num_feats, num_activities, num_layers, seq_length)
    local classifier = nn.Sequential()
    classifier:add(nn.Transpose({2,3}))  -- swap dim2 with dim4 such that input = B x input_size x seq_length x 1
    classifier:add(nn.VolumetricConvolution(input_size, 64, 7,7,7, 2,2,2, 3,3,3))  --64 x 32
    classifier:add(nn.VolumetricBatchNormalization(64, 1e-3))
    classifier:add(nn.ReLU(true))
    classifier:add(Residual(64, 128))
    classifier:add(nn.VolumetricMaxPooling(2,2,2, 2,2,2))  -- 32 -> 16
    classifier:add(Residual(128, num_feats))
    classifier:add(Residual(num_feats, num_feats))
    classifier:add(nn.VolumetricMaxPooling(2,2,2, 2,2,2))  -- 16 -> 8
    classifier:add(Residual(num_feats, num_feats+128))
    classifier:add(Residual(num_feats+128, num_feats+128))
    classifier:add(nn.VolumetricMaxPooling(2,2,2, 2,2,2))  -- 8 -> 4
    classifier:add(Residual(num_feats+128, 512))
    classifier:add(Residual(512, 512))
    classifier:add(Residual(512, 512))
    classifier:add(nn.VolumetricConvolution(512, num_activities, 2,4,4, 1,1,1, 0,0,0))
    classifier:add(nn.View(-1, num_activities))
    return classifier
end

------------------------------------------------------------------------------------------------------------

--[[ Create VGG16 + spatial average pooling + lin layer ]]--
local function create_network()

    local hms_features, params = paths.dofile('../load_hg_best.lua')()
    hms_features:evaluate()

    local classifier = load_classifier_network(params.feat_size,
                                               opt.nFeats,
                                               opt.num_activities,
                                               opt.nLayers,
                                               opt.seq_length)

    return nil, hms_features, classifier, params  -- features, hms, classifier, params
end

------------------------------------------------------------------------------------------------------------

return create_network
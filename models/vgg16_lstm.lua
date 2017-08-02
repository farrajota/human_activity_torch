--[[
    Load VGG16 + LSTM network.
]]


require 'nn'

------------------------------------------------------------------------------------------------------------

local function SelectFeatsDisableBackprop(net)
    local features = net
    features:remove(features:size()) -- remove logsoftmax layer
    features:remove(features:size()) -- remove 3rd linear layer
    return nn.NoBackprop(features)
end

------------------------------------------------------------------------------------------------------------

local function load_network()
    local filepath = paths.concat(projectDir, 'data', 'pretrained_models')

    local net = torch.load(paths.concat(filepath, 'model_vgg16.t7'))
    local params = torch.load(paths.concat(filepath, 'parameters_vgg16.t7'))

    net:remove(net:size()) -- remove logsoftmax layer
    net:remove(net:size()) -- remove 3rd linear layer

    params.feat_size = 4096

    return net, params
end

------------------------------------------------------------------------------------------------------------

--[[ Create VGG16 + LSTM ]]--
local function create_network()
    local vgg16, params = load_network()
    vgg16:evaluate()

    local lstm, view1 = paths.dofile('lstm.lua')(params.feat_size,
                                                 opt.nFeats,
                                                 opt.num_activities,
                                                 opt.nLayers)

    local net = {}
    net.features = vgg16
    net.classifier = lstm
    net.view1 = view1
    net.mean = params.mean
    net.std = params.std
    net.pixel_scale = params.pixel_scale

    function net:forward(input)
        local input2=input:clone()
        local batchSize = input2:size(1)
        local seq_length = input2:size(2)
        input2 = input2:index(3, torch.LongTensor{3,2,1})  -- bgr
        input2 = input2:view(batchSize*seq_length, 3, 224, 224)
        input2:mul(self.pixel_scale)
        for j=1, batchSize*seq_length do
            for i=1, 3 do
                if self.mean then input2[j][i]:add(-self.mean[i]) end
                if self.std then input2[j][i]:div(self.std[i]) end
            end
        end

        local out = self.features:updateOutput(input2)

        self.view1:resetSize(batchSize*seq_length, -1)

        return self.classifier:forward(out:view(batchSize, seq_length, -1))
    end

    function net:backward(input, grads)
        return self.classifier:backward(input, grads)
    end

    function net:cuda()
        self.features:cuda()
        self.classifier:cuda()
    end

    function net:type(dtype)
        self.features:type(dtype)
        self.classifier:type(dtype)
    end

    function net:training()
        self.features:evaluate()
        self.classifier:training()
    end

    function net:evaluate()
        self.features:evaluate()
        self.classifier:evaluate()
    end

    return net
end

------------------------------------------------------------------------------------------------------------

return create_network
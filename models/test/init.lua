--[[
    Network architectures used in tests.
]]

local function additional_network_architectures(network_list)

    local network_list = network_list or {}

    -- LSTM
    network_list['vgg16-lstm'] = paths.dofile('vgg16_lstm.lua')
    network_list['vgg16-lstm2'] = paths.dofile('vgg16_lstm2.lua')
    network_list['vgg16-hms-lstm'] = paths.dofile('vgg16_hms_lstm.lua')
    network_list['hms-lstm'] = paths.dofile('hms_lstm.lua')

    -- ConvNet
    network_list['vgg16-convnet'] = paths.dofile('vgg16_convnet.lua')
    network_list['vgg16-convnet3d'] = paths.dofile('vgg16_convnet3d.lua')
    network_list['vgg16-convnet3d_2'] = paths.dofile('vgg16_convnet3d_2.lua')
    network_list['vgg16-hms-convnet3d'] = paths.dofile('vgg16_hms_convnet3d.lua')
    network_list['vgg16-hms-convnet3d_2'] = paths.dofile('vgg16_hms_convnet3d_2.lua')
    network_list['hms-convnet3d']   = paths.dofile('hms_convnet3d.lua')
    network_list['hms-convnet3d_2'] = paths.dofile('hms_convnet3d_2.lua')
    network_list['hms-convnet3d_3'] = paths.dofile('hms_convnet3d_3.lua')
    network_list['hms-convnet3d_4'] = paths.dofile('hms_convnet3d_4.lua')
    network_list['hms-convnet3d_5'] = paths.dofile('hms_convnet3d_5.lua')
    network_list['hms-convnet3d_6'] = paths.dofile('hms_convnet3d_6.lua')
    network_list['hms-convnet3d_7'] = paths.dofile('hms_convnet3d_7.lua')
    network_list['hms-convnet3d_8'] = paths.dofile('hms_convnet3d_8.lua')
end

return additional_network_architectures
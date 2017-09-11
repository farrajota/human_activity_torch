--[[
    Generate skeletons of annotated persons in a video clip.
]]

require 'torch'
require 'paths'
require 'xlua'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'image'
disp = require 'display'

torch.manualSeed(4)
torch.setdefaulttensortype('torch.FloatTensor')

paths.dofile('../projectdir.lua')
utils = paths.dofile('../util/utils.lua')
paths.dofile('../util/draw.lua')
paths.dofile('../util/heatmaps.lua')
paths.dofile('../data.lua')
paths.dofile('../sample_batch.lua')

local opts = paths.dofile('../options.lua')
opt = opts.parse(arg)
opt.dataset = 'ucf_sports'
opt.rotate = 15
opt.scale = 0.00
opt.rotRate = 0
opt.batchSize = 4
opt.seq_length = 30
opt.same_transform = true
opt.process_input_heatmap = true
opt.process_input_feats = true
opt.use_center_crop = false
niters = 10
mode = 'train'
plot_results = false
model, opt.params = paths.dofile('../models/load_hg_best.lua')()
model =  model.modules[1]
model:cuda()
resnet, resnet_params = paths.dofile('../models/load_resnet.lua')('resnet50')
resnet:remove(resnet:size()) -- remove nn.View
resnet:remove(resnet:size()) -- remove nn.SpatialAveragePool
resnet:cuda()
cudnn.convert(resnet, cudnn)


local function normalize_image(img)
    assert(img)

    -- normalize pixels
    for i=1, 3 do
        if opt.params.mean then img[i]:add(-opt.params.mean[i]) end
        if opt.params.std then img[i]:div(opt.params.std[i]) end
    end

    return img
end


local data_loader = select_dataset_loader(opt.dataset, mode)
local loader = data_loader[mode]

local save_dir = paths.concat(opt.expDir, 'video_skeletons')
if not paths.dirp(save_dir) then
    print('Saving plots to: ' .. save_dir)
    os.execute('mkdir -p ' .. save_dir)
end

print('==> Fetch a video clip from the train data (does not apply any transformation) and detect skeletons')
for i=1, niters do
    local input_hms, imgs_feats, label, imgs_params = getSampleBatch(loader, opt.batchSize, false)

    if not paths.dirp(paths.concat(save_dir, 'iter_' .. i)) then
        print('Saving plots to: ' .. paths.concat(save_dir, 'iter_' .. i))
        os.execute('mkdir -p ' .. paths.concat(save_dir, 'iter_' .. i))
    end

    -- draw skeletons on the image
    local input = torch.Tensor(1,3,256,256):fill(0):cuda()
    for ibatch=1, opt.batchSize do
        local imgs_skeletons = {}
        local imgs = {}
        local crops = {}
        local heatmaps = {}
        local features = {}
        local skeleton_crops = {}
        for j=1, input_hms:size(2) do
            local img = imgs_params[ibatch][j].img
            input[1]:copy(input_hms[ibatch][j])

            -- compute features
            local img_ = image.scale(img, 224, 224)
            for i=1, 3 do
                if resnet_params.mean then img_[i]:add(-resnet_params.mean[i]) end
                if resnet_params.std then img_[i]:div(resnet_params.std[i]) end
            end
            local feats = resnet:forward(img_:view(1,3,224,224):cuda()):float()
            feats = image.scale(feats:squeeze():mean(1):squeeze(), 256,256, 'simple')

            -- compute heatmaps + draw skeleton
            local out = model:forward(input)
            local hms = out[#out]:float()
            hms[hms:lt(0)]=0
            local center = imgs_params[ibatch][j].center
            local scale = imgs_params[ibatch][j].scale
            local preds_hm, preds_img = getPredsFull(hms, center, scale)
            local skeleton = drawSkeleton(img, hms[1], preds_img[1])
            local skeleton_crop = drawSkeleton(input_hms[ibatch][j], hms[1], preds_hm[1]:mul(4))

            -- heatmaps
            local heatmap = image.scale(colorHM(hms[1]:max(1):squeeze()), 256, 256)

            table.insert(imgs_skeletons, skeleton)
            table.insert(imgs, img)
            table.insert(features, feats)
            table.insert(crops, input_hms[ibatch][j])
            table.insert(heatmaps, heatmap)
            table.insert(skeleton_crops, skeleton_crop)
        end

        -- display skeletons
        disp.image(imgs_skeletons, {title='batch'.. i .. '_activity'.. label[ibatch][1]})

        -- save skeletons + imgs to disk
        local save_dir_video = paths.concat(save_dir, 'iter_' .. i, 'video_' .. ibatch)
        if not paths.dirp(save_dir_video) then
            print('Saving plots to: ' .. save_dir_video)
            os.execute('mkdir -p ' .. save_dir_video)
            os.execute('mkdir -p ' .. paths.concat(save_dir_video, 'images'))
            os.execute('mkdir -p ' .. paths.concat(save_dir_video, 'crops'))
            os.execute('mkdir -p ' .. paths.concat(save_dir_video, 'skeletons'))
            os.execute('mkdir -p ' .. paths.concat(save_dir_video, 'features'))
            os.execute('mkdir -p ' .. paths.concat(save_dir_video, 'heatmaps'))
            os.execute('mkdir -p ' .. paths.concat(save_dir_video, 'skeleton_crops'))
        end
        for j=1, #imgs_skeletons do
            local img = imgs[j]
            local skeleton = imgs_skeletons[j]
            local feats = features[j]
            local crop = crops[j]
            local heatmap = heatmaps[j]
            local skeleton_crop = skeleton_crops[j]
            image.save(paths.concat(save_dir_video, 'images', 'img' .. j .. '.png'), img)
            image.save(paths.concat(save_dir_video, 'crops', 'crop' .. j .. '.png'), crop)
            image.save(paths.concat(save_dir_video, 'skeletons', 'skeleton' .. j .. '.png'), skeleton)
            image.save(paths.concat(save_dir_video, 'features', 'feat' .. j .. '.png'), feats)
            image.save(paths.concat(save_dir_video, 'heatmaps', 'heatmap' .. j .. '.png'), heatmap)
            image.save(paths.concat(save_dir_video, 'skeleton_crops', 'skeleton_crop' .. j .. '.png'), skeleton_crop)
        end
    end

    collectgarbage()

    xlua.progress(i, niters)
end

print('Generating skeletons from video clips successfully finished.')
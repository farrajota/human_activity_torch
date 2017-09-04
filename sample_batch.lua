--[[
    Data transform functions.
]]

require 'torch'
require 'image'

paths.dofile('util/img.lua')


------------------------------------------------------------------------------------------------------------

--[[ Function for data augmentation, randomly samples on a normal distribution. ]]--
local function rnd(x)
    return math.max(-2*x, math.min(2*x, torch.randn(1)[1]*x))
end


-------------------------------------------------------------------------------
-- Transform data for one object sample
-------------------------------------------------------------------------------

function transform_data(img, center, scale, params)
    assert(img)
    assert(params)

    -- Crop image
    local img_transf = crop2(img, center, scale, params.rotation, opt.inputRes)

    if not img_transf then return nil end

    -- Flipping
    if params.flip then
        img_transf = flip(img_transf)
    end

    -- color augmentation
    img_transf[1]:mul(params.color_jit[1]):clamp(0,1)
    img_transf[2]:mul(params.color_jit[2]):clamp(0,1)
    img_transf[3]:mul(params.color_jit[3]):clamp(0,1)

    -- output
    return img_transf
end


-------------------------------------------------------------------------------
-- Get a batch of data samples
-------------------------------------------------------------------------------

local function get_random_transforms(is_train)
    assert(is_train ~= nil)
    assert(type(is_train) == 'boolean')

    local rot = 0
    local scale = 1
    local color_jit = {1, 1, 1}
    local flip = false

    if is_train then
        -- scale
        scale = scale * (2 ^ rnd(opt.scale))

        -- rotation
        rot = rnd(opt.rotate)
        if torch.uniform() > opt.rotRate then
            rot = 0
        end

        -- Flipping
        if torch.uniform() > .5 then
            flip = true
        end

        -- color augmentation
        color_jit = {torch.uniform(1-opt.colorjit, 1+opt.colorjit),
                     torch.uniform(1-opt.colorjit, 1+opt.colorjit),
                     torch.uniform(1-opt.colorjit, 1+opt.colorjit)}
    end

    return {
        rotation = rot,
        scale = scale,
        color_jit = color_jit,
        flipping = flip
    }
end

------------------------------------------------------------------------------------------------------------

local function get_center_crop_bbox(img)
    local width, height = img:size(3), img:size(2)
    local pad = math.abs((width - height)/2)
    if width >= height then
        return torch.FloatTensor({pad, 1, pad+height -1, height})
    else
        return torch.FloatTensor({1, pad, width, pad+width -1})
    end
end

------------------------------------------------------------------------------------------------------------

local function normalize_image(img)
    assert(img)

    -- convert to bgr
    if opt.params.colourspace == 'bgr' then
        img = img:index(1, torch.LongTensor{3,2,1})  -- bgr
    end

    -- rescale pixels
    if opt.params.pixel_scale > 1 then
        img:mul(opt.params.pixel_scale)
    end

    -- normalize pixels
    for i=1, 3 do
        if opt.params.mean then img[i]:add(-opt.params.mean[i]) end
        if opt.params.std then img[i]:div(opt.params.std[i]) end
    end

    return img
end

------------------------------------------------------------------------------------------------------------

local function fetch_subset_images(nimgs, seq_length, step)
    assert(nimgs)
    assert(seq_length)
    assert(step)

    local out_idx = {}

    local idx_ini
    if nimgs < seq_length*step then
        idx_ini = math.random(1, nimgs)  -- start randomly (the vid will loop)
    else
        idx_ini = math.random(1, nimgs - seq_length*step)
    end

    table.insert(out_idx, idx_ini)
    local idx_next = idx_ini
    for i=1, seq_length-1 do
        idx_next = idx_next + step
        if idx_next > nimgs then
            idx_next = idx_next - nimgs
        end
        table.insert(out_idx, idx_next)
    end

    return out_idx
end

------------------------------------------------------------------------------------------------------------

local function process_images_heatmaps(imgs, idxs, params_transform, is_test)
    assert(imgs)
    assert(idxs)
    assert(params_transform)

    local is_test = is_test or false

    local imgs_transf = {}

    local prev_center, prev_scale = imgs[1].center, imgs[1].scale
    local prev_bbox = imgs[1].bbox
    if prev_bbox:sum() == 0 or prev_bbox[4]-prev_bbox[2] < 20 then
        prev_bbox = get_center_crop_bbox(imgs[1].img)
        prev_center = torch.FloatTensor({(prev_bbox[1] + prev_bbox[3])/2,
                                        (prev_bbox[2] + prev_bbox[4])/2})
        prev_scale = (prev_bbox[4]-prev_bbox[2]) / 200 * 1.25
    end
    for _, idx in ipairs(idxs) do
        local img = imgs[idx].img
        local scale = imgs[idx].scale
        local center = imgs[idx].center
        local bbox = imgs[idx].bbox

        if scale < 0 or bbox:sum() == 0 or bbox[4]-bbox[2] < 20 then
            bbox = prev_bbox
            center = prev_center
            scale = prev_scale
        end

        if is_test then
            local img_crop = crop2(img, center, scale, 0, opt.inputRes)
            if not img_crop then return {} end -- skip this round of data/transforms if any error occurs
            table.insert(imgs_transf, img_crop)
        else
            if not opt.same_transform then
                params_transform = get_random_transforms(is_train)
            end

            local new_img = transform_data(img, center, scale, params_transform)
            if not new_img then return {} end  -- skip this round of data/transforms if any error occurs
            table.insert(imgs_transf, new_img)
        end

        if not (scale < 0 or bbox:sum() == 0 or bbox[4]-bbox[2] < 20) then
            prev_bbox = bbox
            prev_center = center
            prev_scale = scale
        end
    end

    return imgs_transf
end

------------------------------------------------------------------------------------------------------------

local function process_images_crops(imgs, imgs_transf, idxs, params_transform, is_test)
    assert(imgs)
    assert(idxs)
    assert(params_transform)

    local is_test = is_test or false

    local imgs_resized = {}

    if opt.use_center_crop then
        local iW, iH
        local img_size = torch.random(224,256)
        for i=1, #imgs_transf do
            if is_test then
                local img = imgs_transf[i]
                local new_img = image.scale(img, 224)
                table.insert(imgs_resized, normalize_image(new_img))
            else
                if not opt.same_transform then
                    img_size = torch.random(224,256)
                end

                local new_img = resize_image(imgs_transf[i], img_size)
                if (iW == nil or iH == nil) or not opt.same_transform then
                    iW = torch.random(1, math.max(1, new_img:size(3) - 224))
                    iH = torch.random(1, math.max(1, new_img:size(2) - 224))
                end
                new_img = new_img[{{}, {iH, iH+224 -1}, {iW, iW + 224 -1}}]
                table.insert(imgs_resized, normalize_image(new_img))
            end
        end
    else
        local iW, iH
        local img_size = torch.random(224,256)
        for _, idx in ipairs(idxs) do
            local img = imgs[idx].img
            if is_test then
                new_img = resize_image(img, 224)
                iW = math.max(1, math.floor((new_img:size(3)-224)/2))
                iH = math.max(1, math.floor((new_img:size(2)-224)/2))
            else
                if not opt.same_transform or true then  -- added or true for testing
                    img_size = torch.random(224,256)
                end
                new_img = resize_image(img, img_size)
                if (iW == nil or iH == nil) or not opt.same_transform or true then  -- added or true for testing
                    iW = torch.random(1, math.max(1, new_img:size(3) - 224))
                    iH = torch.random(1, math.max(1, new_img:size(2) - 224))
                end
            end

            new_img = new_img[{{}, {iH, iH+224 -1}, {iW, iW + 224 -1}}]
            table.insert(imgs_resized, normalize_image(new_img))
        end
    end

    return imgs_resized
end

------------------------------------------------------------------------------------------------------------

--[[ Fetch data (images + label) from a single video ]]--
local function fetch_single_data(data_loader, idx, is_train, use_subset)
    assert(data_loader)
    assert(idx)
    assert(is_train ~= nil)
    assert(type(is_train) == 'boolean')
    assert(use_subset ~= nil)

    -- get images + label from a video
    local imgs, label = data_loader.loader(idx)

    -- get cropped images + heatmaps
    local imgs_transf, imgs_resized = {}, {}
    if is_train or use_subset then
        -- select a subset of images from the video
        local idxs = fetch_subset_images(#imgs, opt.seq_length, opt.step)

        -- select some random transformations to apply to the entire set of images
        local params_transform = get_random_transforms(is_train)

        -- process heatmaps
        if opt.process_input_heatmap or opt.use_center_crop then
            imgs_transf = process_images_heatmaps(imgs, idxs, params_transform, false)
        end

        -- process images
        if opt.process_input_feats then
            imgs_resized = process_images_crops(imgs, imgs_transf, idxs, params_transform, false)
        end
    else
        -- select a subset of images from the video
        local idxs = fetch_subset_images(#imgs, opt.test_seq_length, opt.test_step)

        -- process heatmaps
        if opt.process_input_heatmap or opt.use_center_crop then
            imgs_transf = process_images_heatmaps(imgs, idxs, {}, true)
        end

        -- process images
        if opt.process_input_feats then
            imgs_resized = process_images_crops(imgs, imgs_transf, idxs, {}, true)
        end
    end

    imgs = nil
    collectgarbage()

    return {imgs_transf, imgs_resized, label}
end

------------------------------------------------------------------------------------------------------------

--[[ Create/build a batch of images + label ]]--
local function get_batch(data_loader, batchSize, is_train)
    assert(data_loader)
    assert(batchSize)
    assert(is_train ~= nil)
    assert(type(is_train) == 'boolean')

    local size = data_loader.num_activities
    local max_attempts = 10
    local batchData, idxUsed = {}, {}

    for i=1, batchSize do
        local data = {}
        local attempts = 0
        -- select a random activity
        while not next(data) do

            local activity_id = math.random(1, data_loader.num_activities)
            -- select a random video from the selected activity
            local video_ids = data_loader.get_video_ids(activity_id)
            local video_id = video_ids[math.random(1, #video_ids)] + 1  -- set to 1-index

            local idx = video_id
            if not idxUsed[idx] then
                data = fetch_single_data(data_loader, idx, is_train, true)
                idxUsed[idx] = 1

                -- increment attempts counter. This avoids infinite loops
                -- if it cannot find a valid image + annotations
                attempts = attempts + 1

                if attempts > max_attempts then
                    error('Reached the maximum number of attempts to find an unique batch: ' .. max_attempts)
                end
            end

            collectgarbage()
        end
        table.insert(batchData, data)
    end
    collectgarbage()
    return batchData
end

------------------------------------------------------------------------------------------------------------

--[[ Returns a batch of image sequences (used for training). ]]--
function getSampleBatch(data_loader, batchSize, is_train)
    assert(data_loader)
    assert(batchSize)
    assert(is_train ~= nil)
    assert(type(is_train) == 'boolean')

    -- get batch data
    local sample = get_batch(data_loader, batchSize, is_train)

    -- images (for body joints)
    local imgs_hms
    if next(sample[1][1]) then
        imgs_hms = torch.FloatTensor(batchSize, opt.seq_length,
                                           3, opt.inputRes, opt.inputRes):fill(0)
        for i=1, batchSize do
            local prev_sample = sample[i][1][1]
            for j=1, opt.seq_length do
                local sample_ = sample[i][1][j]
                if sample_ then
                    prev_sample = sample_
                else
                    sample_ = prev_sample
                end
                imgs_hms[i][j]:copy(sample_)
            end
        end
    end

    -- images (for body joints)
    local imgs_feats
    if next(sample[1][2]) then
        imgs_feats = torch.FloatTensor(batchSize, opt.seq_length,
                                             3, 224, 224):fill(0)
        for i=1, batchSize do
            local prev_sample = sample[i][2][1]
            for j=1, opt.seq_length do
                local sample_ = sample[i][2][j]
                if sample_ then
                    prev_sample = sample_
                else
                    sample_ = prev_sample
                end
                imgs_feats[i][j]:copy(sample_)
            end
        end
    end

    -- labels
    local labels_tensor = torch.IntTensor(batchSize, opt.seq_length):fill(0)
    for i=1, batchSize do
        labels_tensor[i]:fill(sample[i][3])
    end

    collectgarbage()

    return imgs_hms, imgs_feats, labels_tensor
end


------------------------------------------------------------------------------------------------------------

--[[ Returns a full sequence of images from a video (used for testing). ]]--
function getSampleTest(data_loader, idx)
    assert(data_loader)

    -- get batch data
    local sample = fetch_single_data(data_loader, idx, false, false)

    --[[
    -- sample size
    local sample_seq_length = math.max(#sample[1], #sample[2])
    local min_seq_length = opt.seq_length
    local seq_length = math.max(min_seq_length, sample_seq_length)
    local ini = 0  -- this padding is needed for cases where the test sequence length is smaller than the train seq length
    if min_seq_length > sample_seq_length then
        ini = min_seq_length - sample_seq_length
    end

    -- images data
    --local imgs_hms = torch.FloatTensor(1, seq_length, 3, opt.inputRes, opt.inputRes):fill(0)
    --local imgs_feats = torch.FloatTensor(1, seq_length, 3, 224, 224):fill(0)
    --local ini_id = 1
    --for j=1, seq_length do
    --    imgs_hms[1][j]:copy(sample[1][ini_id])
    --    imgs_feats[1][j]:copy(sample[2][ini_id])
    --    ini_id = ini_id + 1
    --    if ini_id > sample_seq_length then
    --        ini_id = 1
    --    end
    --end

    -- images data
    local imgs_hms, imgs_feats
    if next(sample[1]) then imgs_hms = torch.FloatTensor(1, seq_length, 3, opt.inputRes, opt.inputRes):fill(0) end
    if next(sample[2]) then imgs_feats = torch.FloatTensor(1, seq_length, 3, 224, 224):fill(0) end
    local ini_id = 1
    for j=1, seq_length do
        if imgs_hms then imgs_hms[1][j]:copy(sample[1][ini_id]) end
        if imgs_feats then imgs_feats[1][j]:copy(sample[2][ini_id]) end
        ini_id = ini_id + 1
        if ini_id > sample_seq_length then
            ini_id = 1
        end
    end
    ]]

    local seq_length = math.max(#sample[1], #sample[2])
    local imgs_hms, imgs_feats
    if next(sample[1]) then imgs_hms = torch.FloatTensor(1, seq_length, 3, opt.inputRes, opt.inputRes):fill(0) end
    if next(sample[2]) then imgs_feats = torch.FloatTensor(1, seq_length, 3, 224, 224):fill(0) end
    for i=1, seq_length do
        if imgs_hms then imgs_hms[1][i]:copy(sample[1][i]) end
        if imgs_feats then imgs_feats[1][i]:copy(sample[2][i]) end
    end

    -- labels
    local labels_tensor = torch.IntTensor(1, seq_length):fill(sample[3])

    collectgarbage()

    return imgs_hms, imgs_feats, labels_tensor
end

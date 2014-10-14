require 'image'

-- data augmentation methods

local function crop(x, offsets, width, height)
   height = height or width
   return image.crop(x, offsets[1], offsets[2], offsets[1] + width, offsets[2] + height)
end
local function horizontal_reflection(x)
   return image.hflip(x)
end
local function zoomout(x)
   return image.scale(x, 24, 24, 'bilinear')
end
local CROP_POS24 = {}
local function generate_crop_pos24()
   for i = 0, 8, 4 do
      for j = 0, 8, 4 do
	 table.insert(CROP_POS24, {i, j})
      end
   end
end
local CROP_POS28 = {}
local function generate_crop_pos28()
   for i = 0, 4, 2 do
      for j = 0, 4, 2 do
	 table.insert(CROP_POS28, {i, j})
      end
   end
end
local CROP_POS30 = {}
local function generate_crop_pos30()
   for i = 0, 2, 2 do
      for j = 0, 2, 2 do
	 table.insert(CROP_POS30, {i, j})
      end
   end
end
generate_crop_pos24()
generate_crop_pos28()
generate_crop_pos30()

function data_augmentation(x, y)
   local scale = #CROP_POS24 + #CROP_POS28 + #CROP_POS30
   if x:dim() == 4 then
      -- jitter for training
      local new_x = torch.Tensor(x:size(1) * scale * 2,
				 3, 24, 24)
      local new_y = torch.Tensor(y:size(1) * scale * 2,
				 y:size(2))
      for i = 1, x:size(1) do
	 local src = x[i]
	 local images = {}
	 for j = 1, #CROP_POS24 do
	    table.insert(images, crop(src, CROP_POS24[j], 24))
	 end
	 for j = 1, #CROP_POS28 do
	    table.insert(images, zoomout(crop(src, CROP_POS28[j], 28)))
	 end
	 for j = 1, #CROP_POS30 do
	    table.insert(images, zoomout(crop(src, CROP_POS30[j], 30)))
	 end
	 for j = 1, #images do
	    new_x[scale * 2 * (i - 1) + j]:copy(images[j])
	    new_y[scale * 2 * (i - 1) + j]:copy(y[i])
	    new_x[scale * 2 * (i - 1) + #images + j]:copy(horizontal_reflection(images[j]))
	    new_y[scale * 2 * (i - 1) + #images + j]:copy(y[i])
	 end
	 if i % 100 == 0 then
	    collectgarbage()
	 end
      end
      return new_x, new_y
   elseif x:dim() == 3 then
      -- jitter for prediction
      local new_x = torch.Tensor(scale * 2, 3, 24, 24)
      local src = x
      local images = {}
      for i = 1, #CROP_POS24 do
	 table.insert(images, crop(src, CROP_POS24[i], 24))
      end
      for i = 1, #CROP_POS28 do
	 table.insert(images, zoomout(crop(src, CROP_POS28[i], 28)))
      end
      for i = 1, #CROP_POS30 do
	 table.insert(images, zoomout(crop(src, CROP_POS30[i], 30)))
      end
      for i = 1, #images do
	 new_x[i]:copy(images[i])
	 new_x[#images + i]:copy(horizontal_reflection(images[i]))
      end
      return new_x
   end
end
--[[
require './save_images'
local x = torch.load("../data/train_x.bin")
local y = torch.load("../data/train_y.bin")
local target = 13
local jit = data_augmentation(x[target]:resize(1, 3, 32, 32), y[target]:resize(1, 10))
print(jit:size(1))
save_images(jit, jit:size(1), "jitter.png")
--]]
return true

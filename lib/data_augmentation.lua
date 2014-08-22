require 'image'

-- data augmentation methods

local function crop(x, offsets, size)
   return image.crop(x, offsets[1], offsets[2], offsets[1] + size, offsets[2] + size)
end
local function horizontal_reflection(x)
   return image.hflip(x)
end
local function zoomout(x)
   return image.scale(x, 24, 24, 'bilinear')
end
local CROP_POS24 = {}
function generate_crop_pos24()
   for i = 0, 8, 4 do
      for j = 0, 8, 4 do
	 table.insert(CROP_POS24, {i, j})
      end
   end
end
local CROP_POS28 = {}
function generate_crop_pos28()
   for i = 0, 4, 2 do
      for j = 0, 4, 2 do
	 table.insert(CROP_POS28, {i, j})
      end
   end
end
generate_crop_pos24()
generate_crop_pos28()

local function random_crop(x, scale)
   local index = torch.randperm(#CROP_POS24)
   local images = {}
   for i = 1, scale do
      table.insert(images, crop(x, CROP_POS24[index[i]], 24))
   end
   return images
end
function data_augmentation(x, y)
   local scale = 9 + 9
   if x:dim() == 4 then
      -- jitter for training
      local new_x = torch.Tensor(x:size(1) * scale * 2,
				 3, 24, 24)
      local new_y = torch.Tensor(y:size(1) * scale * 2,
				 y:size(2))
      for i = 1, x:size(1) do
	 local src = x[i]
	 local images = {}
	 for i = 1, #CROP_POS24 do
	    table.insert(images, crop(src, CROP_POS24[i], 24))
	 end
	 for i = 1, #CROP_POS28 do
	    table.insert(images, zoomout(crop(src, CROP_POS28[i], 28)))
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
      local new_x = torch.Tensor(18 * 2, 3, 24, 24)
      local src = x
      local images = {}
      for i = 1, #CROP_POS24 do
	 table.insert(images, crop(src, CROP_POS24[i], 24))
      end
      for i = 1, #CROP_POS28 do
	 table.insert(images, zoomout(crop(src, CROP_POS28[i], 28)))
      end
      for i = 1, #images do
	 new_x[i]:copy(images[i])
	 new_x[#images + i]:copy(horizontal_reflection(images[i]))
      end
      return new_x
   end
end

--require './save_images'
--local data = torch.load("../data/train_x.bin")
--local jit = data_augmentation(data[13])
--save_images(jit, 10, "jitter.png")

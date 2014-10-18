require 'cutorch'
require 'cunn'
require './SETTINGS'
require './lib/data_augmentation'
require './lib/preprocessing'
require './nin_model.lua'

-- multiple model averaging
-- (same trainset, different initial weights, different minibatch order)

local function predict(file, models, params, test_x)
   local BATCH_SIZE = 100
   local DA_SIZE = nil
   local fp = io.open(file, "w")
   local preds = torch.Tensor(test_x:size(1), 10):zero()
   if test_x:size(1) % BATCH_SIZE ~= 0 then
      error("expect test size % " .. BATCH_SIZE .. " == 0")
   end
   fp:write("id,label\n")
   for i = 1, test_x:size(1), BATCH_SIZE do
      local step = 64
      if DA_SIZE == nil then
	 local test_da = data_augmentation(test_x[1])
	 DA_SIZE = test_da:size()
      end
      local x = torch.Tensor(BATCH_SIZE, DA_SIZE[1], DA_SIZE[2], DA_SIZE[3], DA_SIZE[4])
      local index = torch.LongTensor(BATCH_SIZE, DA_SIZE[1])
      for j = 1, BATCH_SIZE do
         x[j]:copy(data_augmentation(test_x[i + j - 1]))
	 index[j]:fill(i + j - 1)
      end
      x = x:view(BATCH_SIZE * DA_SIZE[1], 3, 24, 24)
      index = index:view(BATCH_SIZE * DA_SIZE[1])
      preprocessing(x, params)
      x = x:cuda()
      for j = 1, x:size(1), step do
	 local batch = torch.Tensor(step, x:size(2), x:size(3), x:size(4)):zero()
	 local n = step
	 if j + n > x:size(1) then
	    n = 1 + n - ((j + n) - x:size(1))
	 end
	 batch:narrow(1, 1, n):copy(x:narrow(1, j, n))
	 batch = batch:cuda()
	 for k = 1, #models do
	    local z = models[k]:forward(batch):float()
	    -- averaging
	    for l = 1, n do
	       preds[index[j + l -1]] = preds[index[j + l -1]] + z[l]
	    end
	 end
      end
      for j = 1, BATCH_SIZE do
         local max_v, max_i = preds[i + j - 1]:max(1)
         fp:write(string.format("%d,%s\n", i + j -1, ID2LABEL[max_i[1]]))
      end
      xlua.progress(i, test_x:size(1))
      collectgarbage()
   end
   xlua.progress(test_x:size(1), test_x:size(1))
   fp:close()
end
local function prediction()
   local x = torch.load(string.format("%s/test_x.bin", DATA_DIR))
   
   -- models
   local models = {
      torch.load("ec2/node1/very_deep_20.model"):cuda(),
      torch.load("ec2/node2/very_deep_20.model"):cuda(),
      torch.load("ec2/node3/very_deep_20.model"):cuda(),
      torch.load("ec2/node4/very_deep_20.model"):cuda(),
      torch.load("ec2/node5/very_deep_20.model"):cuda(),
      torch.load("ec2/node6/very_deep_20.model"):cuda()
   }
   local params = torch.load("ec2/node1/preprocessing_params.bin")

   for i = 1, #models do
      models[i]:evaluate()
   end
   
   predict("./submission_6model.txt", models, params, x)
end

prediction()

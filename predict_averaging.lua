require 'cutorch'
require 'cunn'
require './SETTINGS'
require './lib/data_augmentation'
require './lib/preprocessing'
require './nin_model.lua'

-- multiple model averaging
-- (same trainset, different initial weights, different minibatch order)

local function predict(file, models, params, test_x)
   local fp = io.open(file, "w")
   fp:write("id,label\n")
   for i = 1, test_x:size(1) do
      local preds = torch.Tensor(10):zero()
      local x = data_augmentation(test_x[i])
      local step = 64
      preprocessing(x, params)
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
	       preds = preds + z[l]
	    end
	 end
      end
      preds:div(x:size(1) * #models)
      
      local max_v, max_i = preds:max(1)
      fp:write(string.format("%d,%s\n", i, ID2LABEL[max_i[1]]))
      xlua.progress(i, test_x:size(1))
      if i % 1000 == 0 then
	 collectgarbage()
      end
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
   
   predict("./submission_6model.txt", models, params, x)
end

prediction()

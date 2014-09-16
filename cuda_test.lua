require 'cutorch'
require 'cunn'
require 'ccn2'
require './very_deep_model'

local function cuda_test()
   local model = very_deep_model():cuda()
   local criterion = nn.MSECriterion():cuda()
   local x = torch.Tensor(64, 3, 24, 24):uniform():cuda()
   local y = torch.Tensor(64, 10):bernoulli(0.1):cuda()
   local z = model:forward(x)
   local df_do = torch.Tensor(z:size(1), y:size(2)):zero()
   for i = 1, z:size(1) do
      local err = criterion:forward(z[i], y[i])
      df_do[i]:copy(criterion:backward(z[i], y[i]))
   end
   model:backward(x, df_do:cuda())
   print("CUDA Test Successful!")
end

torch.setdefaulttensortype('torch.FloatTensor')
print(cutorch.getDeviceProperties(cutorch.getDevice()))
cuda_test()

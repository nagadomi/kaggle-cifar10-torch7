require 'cunn'
require 'ccn2'

-- Inception Architecture (aka GoogLeNet)

-- inception 1x1,1x1+3x3,1x1+5x5,poolproj module
function inception_module(depth_dim, input_size, config)
   local conv1 = nil   
   local conv3 = nil
   local conv5 = nil
   local pool = nil
   
   local depth_concat = nn.DepthConcat(depth_dim)
   conv1 = nn.Sequential()
   conv1:add(nn.SpatialConvolutionMM(input_size, config[1][1], 1, 1))
   conv1:add(nn.ReLU())
   depth_concat:add(conv1)
   conv3 = nn.Sequential()
   conv3:add(nn.SpatialConvolutionMM(input_size, config[2][1], 1, 1))
   conv3:add(nn.ReLU())
   conv3:add(nn.SpatialConvolutionMM(config[2][1], config[2][2], 3, 3))
   conv3:add(nn.ReLU())
   depth_concat:add(conv3)
   conv5 = nn.Sequential()
   conv5:add(nn.SpatialConvolutionMM(input_size, config[3][1], 1, 1))
   conv5:add(nn.ReLU())
   conv5:add(nn.SpatialConvolutionMM(config[3][1], config[3][2], 5, 5))
   conv5:add(nn.ReLU())
   depth_concat:add(conv5)
   pool = nn.Sequential()
   pool:add(nn.SpatialMaxPooling(config[4][1], config[4][1], 1, 1))
   pool:add(nn.SpatialConvolutionMM(input_size, config[4][2], 1, 1))
   pool:add(nn.ReLU())
   depth_concat:add(pool)
   
   return depth_concat
end
function inception_model() -- validate.lua Acc:
   local model = nn.Sequential() 
   
   -- first convolution layer (VGG configuration)

   model:add(nn.SpatialConvolutionMM(3, 64, 3, 3, 1, 1, 1))
   model:add(nn.ReLU())
   model:add(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1, 1))
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

   -- inception 3a
   model:add(inception_module(2, 64, {{64}, {96, 128}, {16, 32}, {3, 32}}))

   -- inception 3b
   model:add(inception_module(2, 256, {{128}, {128, 192}, {32, 96}, {3, 64}}))

   -- maxpool
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

   -- inception 4a
   model:add(inception_module(2, 480, {{192}, {96, 208}, {16, 48}, {3, 64}}))

   -- inception 4b
   model:add(inception_module(2, 512, {{160}, {112, 224}, {24, 64}, {3, 64}}))

   -- inception 4c
   model:add(inception_module(2, 512, {{128}, {128, 256}, {24, 64}, {3, 64}}))

   -- maxpool
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
   model:add(nn.Dropout(0.4))

   model:add(nn.SpatialConvolutionMM(512, 10, 3, 3, 1, 1))
   model:add(nn.Reshape(10))
   model:add(nn.SoftMax())
   
   return model
end
--[[
model = inception_model()
model:cuda()
x = torch.Tensor(64, 3, 24, 24):uniform():cuda()
z = model:forward(x)
print(z:size())
print(model:backward(x, z):size())
--]]
   

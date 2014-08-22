require 'cunn'
require './lib/SpatialAveragePooling'

-- Network in Network model
function nin_model() -- validate.lua Acc: 0.911
   local model = nn.Sequential()
   local final_mlpconv_layer = nil
   
   -- MLP Convolution Layers
   
   model:add(nn.SpatialZeroPadding(2, 2, 2, 2))
   model:add(nn.SpatialConvolutionMM(3, 128, 5, 5, 1, 1))
   model:add(nn.ReLU())
   model:add(nn.SpatialConvolutionMM(128, 96, 1, 1, 1))
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
   model:add(nn.Dropout(0.25))
   
   model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
   model:add(nn.SpatialConvolutionMM(96, 192, 5, 5, 1, 1))
   model:add(nn.ReLU())
   model:add(nn.SpatialConvolutionMM(192, 256, 1, 1, 1, 1))
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
   model:add(nn.Dropout(0.5))
   
   model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
   model:add(nn.SpatialConvolutionMM(256, 256, 5, 5, 1, 1))
   model:add(nn.ReLU())
   model:add(nn.SpatialConvolutionMM(256, 1024, 1, 1, 1, 1))
   model:add(nn.ReLU())
   
   -- Global Average Pooling Layer
   
   final_mlpconv_layer = nn.SpatialConvolutionMM(1024, 10, 1, 1, 1, 1)
   model:add(final_mlpconv_layer)
   model:add(nn.ReLU())
   model:add(nn.SpatialAveragePooling(10, 3, 3, 3, 3))
   model:add(nn.Reshape(10))
   model:add(nn.SoftMax())
   
   -- all initial values in final layer must be a positive number.
   -- this trick is awfully important ('-')b
   final_mlpconv_layer.weight:abs()
   final_mlpconv_layer.bias:abs()
   
   return model
end

require 'cunn'
require 'ccn2'
require './lib/SpatialAveragePooling'

-- Network in Network model
function nin_model() -- validate.lua Acc: 0.911
   local model = nn.Sequential() 
   local final_mlpconv_layer = nil
   
   -- MLP Convolution Layers
   
   model:add(nn.Transpose({1,4},{1,3},{1,2}))
   
   model:add(ccn2.SpatialConvolution(3, 128, 5, 1, 3))
   model:add(nn.ReLU())
   model:add(ccn2.SpatialConvolution(128, 96, 1, 1))
   model:add(nn.ReLU())
   model:add(ccn2.SpatialMaxPooling(3, 2))
   model:add(nn.Dropout(0.25))
   
   model:add(ccn2.SpatialConvolution(96, 192, 5, 1, 2))
   model:add(nn.ReLU())
   model:add(ccn2.SpatialConvolution(192, 256, 1, 1))
   model:add(nn.ReLU())
   model:add(ccn2.SpatialMaxPooling(3, 2))
   model:add(nn.Dropout(0.5))
   
   model:add(ccn2.SpatialConvolution(256, 256, 5, 1, 2))
   model:add(nn.ReLU())
   model:add(ccn2.SpatialConvolution(256, 1024, 1, 1))
   model:add(nn.ReLU())

   -- Global Average Pooling Layer
   
   model:add(nn.Transpose({4,1},{4,2},{4,3}))
   final_mlpconv_layer = nn.SpatialConvolutionMM(1024, 10, 1, 1, 1, 1)
   model:add(final_mlpconv_layer)
   model:add(nn.ReLU())
   model:add(nn.MySpatialAveragePooling(10, 5, 5, 5, 5))
   model:add(nn.Reshape(10))
   model:add(nn.SoftMax())

   -- all initial values in final layer must be a positive number.
   -- this trick is awfully important ('-')b
   final_mlpconv_layer.weight:abs()
   final_mlpconv_layer.bias:abs()
   
   return model
end

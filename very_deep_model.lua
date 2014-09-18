require 'cunn'
require 'ccn2'

-- Very Deep model
function very_deep_model() -- validate.lua Acc: 0.924
   local model = nn.Sequential() 
   local final_mlpconv_layer = nil
   
   -- Convolution Layers
   
   model:add(nn.Transpose({1,4},{1,3},{1,2}))
   model:add(ccn2.SpatialConvolution(3, 64, 3, 1, 1))
   model:add(nn.ReLU())
   model:add(ccn2.SpatialConvolution(64, 64, 3, 1, 1))
   model:add(nn.ReLU())
   model:add(ccn2.SpatialMaxPooling(2, 2))
   model:add(nn.Dropout(0.25))
      
   model:add(ccn2.SpatialConvolution(64, 128, 3, 1, 1))
   model:add(nn.ReLU())
   model:add(ccn2.SpatialConvolution(128, 128, 3, 1, 1))
   model:add(nn.ReLU())
   model:add(ccn2.SpatialMaxPooling(2, 2))
   model:add(nn.Dropout(0.25))
   
   model:add(ccn2.SpatialConvolution(128, 256, 3, 1, 1))
   model:add(nn.ReLU())
   model:add(ccn2.SpatialConvolution(256, 256, 3, 1, 1))
   model:add(nn.ReLU())
   model:add(ccn2.SpatialConvolution(256, 256, 3, 1, 1))
   model:add(nn.ReLU())
   model:add(ccn2.SpatialConvolution(256, 256, 3, 1, 1))
   model:add(nn.ReLU())
   model:add(ccn2.SpatialMaxPooling(2, 2))
   model:add(nn.Dropout(0.25))
   
   -- Fully Connected Layers   
   model:add(ccn2.SpatialConvolution(256, 1024, 3, 1, 0))
   model:add(nn.ReLU())
   model:add(nn.Dropout(0.5))
   model:add(ccn2.SpatialConvolution(1024, 1024, 1, 1, 0))
   model:add(nn.ReLU())
   model:add(nn.Dropout(0.5))
   
   model:add(nn.Transpose({4,1},{4,2},{4,3}))
   
   model:add(nn.SpatialConvolutionMM(1024, 10, 1, 1))
   model:add(nn.Reshape(10))
   model:add(nn.SoftMax())

   return model
end

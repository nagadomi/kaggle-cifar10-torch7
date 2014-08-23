require 'image'
require './SETTINGS'

string.split_it = function(str, sep)
   if str == nil then return nil end
   return string.gmatch(str, "[^\\" .. sep .. "]+")
end
string.split = function(str, sep)
   local ret = {}
   for seg in string.split_it(str, sep) do
      ret[#ret+1] = seg
   end
   return ret
end
local function label_vector(label_name)
   local vec = torch.Tensor(10):zero()
   vec[LABEL2ID[label_name]] = 1.0
   return vec
end
local TRAIN_N = 50000
local function convert_train()
   local label_file = string.format("%s/trainLabels.csv", DATA_DIR)
   local x = torch.Tensor(TRAIN_N, 3, 32, 32)
   local y = torch.Tensor(TRAIN_N, 10)
   local file = io.open(label_file, "r")
   local head = true
   local line
   local i = 1
   for line in file:lines() do
      if head then
	 head = false
      else
	 local col = string.split(line, ",")
	 local img = image.load(string.format("%s/train/%d.png", DATA_DIR, tonumber(col[1])))
	 x[i]:copy(img)
	 y[i]:copy(label_vector(col[2]))
	 if i % 100 == 0 then
	    xlua.progress(i, TRAIN_N)
	 end
	 i = i + 1
      end
   end
   file:close()
   
   torch.save(string.format("%s/train_x.bin", DATA_DIR), x)
   torch.save(string.format("%s/train_y.bin", DATA_DIR), y)
end
local TEST_N = 300000
local function convert_test()
   print(label_file)
   local x = torch.Tensor(TEST_N, 3, 32, 32)
   local i = 1
   for i = 1, TEST_N do
      local img = image.load(string.format("%s/test/%d.png", DATA_DIR, i))
      x[i]:copy(img)
      if i % 100 == 0 then
	 xlua.progress(i, TEST_N)
      end
   end
   torch.save(string.format("%s/test_x.bin", DATA_DIR), x)
end

print("convert train data ...")
convert_train()
print("convert test data ...")
convert_test()

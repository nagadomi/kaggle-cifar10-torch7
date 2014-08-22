require 'torch'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)

DATA_DIR="./data"

CLASSES = {}
LABEL2ID = {
   frog = 1,
   truck = 2,
   deer = 3,
   automobile = 4,
   bird = 5,
   horse = 6,
   ship = 7,
   cat = 8,
   dog = 9,
   airplane = 10
}
ID2LABEL = {}
for k, v in pairs(LABEL2ID) do
   ID2LABEL[v] = k
   CLASSES[v] = k
end

return true

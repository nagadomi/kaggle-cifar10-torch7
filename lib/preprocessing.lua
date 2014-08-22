require 'torch'
require 'image'

-- memory efficient version of unsup.zca_whiten/unsup.pcacov.
-- original version is can be found at: https://github.com/koraykv/unsup/
local function pcacov(x)
   local mean = torch.mean(x, 1)
   local xm = x:clone()
   for i = 1, xm:size(1) do
      xm[i]:add(-1, mean)
   end
   local c = torch.mm(xm:t(), xm)
   c:div(x:size(1)-1)
   local ce,cv = torch.symeig(c,'V')
   return ce,cv
end
local function zca_whiten(data, means, P, invP, epsilon)
    local epsilon = epsilon or 1e-5
    local data_size = data:size()
    local dims = data:size()
    local nsamples = dims[1]
    local n_dimensions = data:nElement() / nsamples
    if data:dim() >= 3 then
       data:resize(nsamples, n_dimensions)
    end
    if not means or not P or not invP then 
        -- compute mean vector if not provided 
       means = torch.mean(data, 1):squeeze()
        -- compute transformation matrix P if not provided
       local ce, cv = pcacov(data)
       collectgarbage()
       ce:add(epsilon):sqrt()
       local invce = ce:clone():pow(-1)
       local invdiag = torch.diag(invce)
       P = torch.mm(cv, invdiag)
       P = torch.mm(P, cv:t())

        -- compute inverse of the transformation
       local diag = torch.diag(ce)
       invP = torch.mm(cv, diag)
       invP = torch.mm(invP, cv:t())
    end
    -- remove the means
    for i = 1, data:size(1) do
       data[i]:add(-1, means)
    end
    -- transform in ZCA space
    data:copy(torch.mm(data, P))
    data:resize(data_size)
    collectgarbage()
    
    return data, means, P, invP
end
local function zca(x, means, P, invP)
   local ax
   ax, means, P, invP = zca_whiten(x, means, P, invP, 0.01) -- 0.1
   x:copy(ax)
   return means, P, invP
end
local function global_contrast_normalization(x, mean, std)
   local scale = 1.0
   local u = mean or x:mean(1)
   local v = std or (x:std(1):div(scale))
   for i = 1, x:size(1) do
      x[i]:add(-u)
      x[i]:cdiv(v)
   end
   return u, v
end
function preprocessing(x, params)
   params = params or {}
   params['gcn_mean'], params['gcn_std'] = global_contrast_normalization(x, params['gcn_mean'], params['gcn_std'])
   params['zca_u'], params['zca_p'], params['zca_invp'] = zca(x, params['zca_u'], params['zca_p'], params['zca_invp'])
   
   return params
end

return true

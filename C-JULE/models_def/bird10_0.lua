 require 'nn'
require 'cunn'

local backend_name = 'nn'

local backend
if backend_name == 'cudnn' then
  require 'cudnn'
  backend = cudnn
else
  backend = nn
end
  
local net = nn.Sequential()

-- building block
local function ConvBNReLU(module, nInputPlane, nOutputPlane)
  module:add(backend.SpatialConvolution(nInputPlane, nOutputPlane, 7, 7, 1, 1, 3, 3))
  module:add(nn.SpatialBatchNormalization(nOutputPlane))
  module:add(backend.ReLU(true))
  --module:add(backend.ReLU6(true))
  --module:add(backend.PReLU(nOutputPlane))
  --module:add(backend.ELU(0.2, true))
  --module:add(backend.LeakyReLU(true, true))
  return module
end

-- Will use "ceil" MaxPooling because we want to save as much
-- space as we can
local MaxPooling = backend.SpatialMaxPooling

-- containing multiple sequentials
local nseqs = 3
local nInputPlanes = {1,32,64}
local nOutputPlanes = {32,64,128}
for i = 1, nseqs do
  module = nn.Sequential()
  ConvBNReLU(module, nInputPlanes[i], nOutputPlanes[i])
  module:add(MaxPooling(2,2,2,2):ceil())
  net:add(module)
end
-- In the last block of convolutions the inputs are smaller than
-- the kernels and cudnn doesn't handle that, have to use cunn
backend = nn
--net:add(nn.Tanh())
net:add(nn.View(10*4*128))
net:add(nn.Linear(10*4*128,160))
net:add(nn.Normalize(2))

return net

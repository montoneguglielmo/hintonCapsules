from layers   import *
from networks import *
from torch    import nn


class netCaps(netClass):
    
    def __init__(self, **kwargs):
        super(netCaps, self).__init__(**kwargs)

        if 'stdWfc' in kwargs:
            stdvW = kwargs['stdWfc']
            self.stdvW = stdvW

        self.lst_modules.append(simpleConv(in_channels=1, out_channels=256, kernel_size=9, stride=1))
        self.lst_modules.append(startCapsuleLayer(dim_out_capsules=8, n_inp_filters=256, n_capsules=32, kernel_size=9, stride=2))
        
        self.lst_modules.append(fcCapsuleLayer(n_out_caps=10, n_inp_caps=32 * 6 * 6, dim_inp_capsules=8, dim_out_capsules=16))

        stdv = np.sqrt(float(self.stdvW))
        self.lst_modules[2].route_weights.data.uniform_(-stdv, stdv)

        
    def forward(self, x):
        x = self.lst_modules[0](x)
        x = F.relu(x, inplace=True)
        x1 = self.lst_modules[1](x)
        x  = x1.view(x1.shape[0], x1.shape[1]*x1.shape[2] * x1.shape[3], x1.shape[4])
        x  = self.lst_modules[2](x)
        return x, x1

        
    def getstatejson(self):
        n_params = 0
        state = {}
        for module in self.lst_modules:
            name = module.__class__.__name__
            n_params += module.printInfo()
            state[name] = module.getstatejson()
        state['n_total_params'] = n_params
        state['stdWfc'] = self.stdvW
        return state


    def post_process(self):
        for module in self.lst_modules:
            module.post_process()

if __name__ == "__main__":

    cnet = netCaps(stdWfc=1e5)

    input = torch.randn(5, 1, 28, 28)
    input = Variable(input)
    output = cnet(input)
    print output.shape
    
    
    print cnet.getstatejson()

    

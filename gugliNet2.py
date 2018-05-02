from layers   import *
from networks import *
from torch    import nn


class netCaps(netClass):
    
    def __init__(self, **kwargs):
        super(netCaps, self).__init__(**kwargs)

        if 'stdWfc' in kwargs:
            stdvW = kwargs['stdWfc']
            self.stdvW = stdvW

        if 'stdWconv' in kwargs:
            self.stdWconv = kwargs['stdWconv']

        self.lst_modules.append(simpleConv(in_channels=1, out_channels=256, kernel_size=5, stride=1))

        #dimension output 24
        self.lst_modules.append(startCapsuleLayer(dim_out_capsules=8, n_inp_filters=256, n_capsules=32, kernel_size=4, stride=1))
        
        #dimension output 21
        self.lst_modules.append(convCapsuleLayer(n_flt_inp=32, n_flt_out=32, inp_sz=21, dim_inp_caps=8, dim_out_caps=12, flt_sz=3, num_iterations=3))
        
        #dimensions output 7
        self.lst_modules.append(fcCapsuleLayer(n_out_caps=10, n_inp_caps=32 * 7 * 7, dim_inp_capsules=12, dim_out_capsules=16))

        stdv      = np.sqrt(float(self.stdvW))
        stdvConv  = np.sqrt(float(self.stdWconv))
        self.lst_modules[3].route_weights.data.uniform_(-stdv, stdv)
        self.lst_modules[2].route_weights.data.uniform_(-stdvConv, stdvConv)
        #self.lst_modules[1].route_weights.data.uniform_(-stdvConv, stdvConv)
        
    def forward(self, x):
        x = self.lst_modules[0](x)
        x = self.lst_modules[1](x)
        x = self.lst_modules[2](x)
        x = x.view(x.shape[0], x.shape[1]*x.shape[2] * x.shape[3], x.shape[4])
        x = self.lst_modules[3](x)
        return x

        
    def getstatejson(self):
        n_params = 0
        state = {}
        for module in self.lst_modules:
            name = module.__class__.__name__
            n_params += module.printInfo()
            state[name] = module.getstatejson()
        state['n_total_params'] = n_params
        state['stdWfc']   = self.stdvW
        state['stdWconv'] = self.stdWconv
        return state


    def post_process(self):
        for module in self.lst_modules:
            module.post_process()



if __name__ == "__main__":

    cnet = netCaps(stdWfc=1e5, stdWconv=1.0)

    input  = torch.randn(5, 1, 28, 28)
    input  = Variable(input)
    output = cnet(input)
    print output.shape
    
    
    print cnet.getstatejson()

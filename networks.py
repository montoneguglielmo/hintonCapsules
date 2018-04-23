from torch import nn



class netClass(nn.Module):
    
    def __init__(self, **kwargs):
        super(netClass, self).__init__()
        self.lst_modules = nn.ModuleList()
        
    def forward(self, x):
        return x

    def post_process(self):
        for module in self.lst_modules:
            module.post_process()

    def getstatejson(self):
        n_params = 0
        state = {}
        for module in self.lst_modules:
            name = module.__class__.__name__
            n_params += module.printInfo()
            state[name] = module.getstatejson()
        state['n_total_params'] = n_params
        return state





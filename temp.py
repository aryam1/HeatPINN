import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "plotly_mimetype+notebook"

class Network(nn.Module):

  def __init__(self, input, width, depth, output):
    super().__init__()

    # activation function
    activation = nn.ReLU

    # input layer
    self.input = nn.Sequential(*[nn.Linear(input, depth), activation()])

    # hidden layers
    self.hidden = nn.Sequential(*[nn.Sequential(*[nn.Linear(depth, depth), activation()]) for _ in range(width-2)])

    # output layer
    self.output = nn.Linear(depth, output)

    self.constants()

  def constants(self):
    self.L = 0.1 # m

    self.cp = 0.7 # J/gK
    self.p = 8000 # kg/m^3
    self.k = 15 # W/mK

    self.a = self.k/(self.p*self.cp) # m^2/s
    
    self.t_ref_t = 1473 # K
    self.t_ref_b = 1273 # K

    self.ht = 200
    self.hb = 100

    self.Bi_t =  self.ht*self.L/self.k
    self.Bi_b = self.hb*self.L/self.k   
    
    self.lossTotal = None
    
  def forward(self, x, t):
    inp = torch.cat((x, t), dim = 1)
    inp = self.input(inp)
    inp = self.hidden(inp)
    inp = self.output(inp)
    return inp
  
  def init_boundaries(self, N):

    # x and t boundary points for training x boundaries, x = -1,1, varying t values
    minusx_boundary = torch.tensor(-1.).view(-1, 1).requires_grad_(True).to(device)
    x_boundary = torch.tensor(1.).view(-1, 1).requires_grad_(True).to(device)
    t_boundary = 10*torch.rand(N).view(-1, 1).requires_grad_(True).to(device)

    # 0 time boundary for training
    x_IC = (-2*torch.rand(N)+1).view(-1, 1).requires_grad_(True).to(device) # X = -1,1 random values
    t_IC = torch.tensor(0.).view(-1, 1).requires_grad_(True).to(device) # T = 0
    
    self.BCpoints = [minusx_boundary, x_boundary, t_boundary]
    self.ICs = [x_IC, t_IC]
  
  def sample(self, N):
      
      # sample points for training
      x = (-2*torch.rand(N)+1).view(-1, 1).requires_grad_(True).to(device)
      t = 10*torch.rand(N).view(-1, 1).requires_grad_(True).to(device)
  
      return x, t

  def derivative(self,f,x):
    return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f).to(device), create_graph=True)[0]
  
  def losses(self,N):

    self.init_boundaries(N)
    x_points, t_points = self.sample(N)

    # losses for x and t boundaries
    minusx_loss = torch.empty(0).to(device)
    x_loss = torch.empty(0).to(device)
    
    for t in self.BCpoints[2]:
      t = t.unsqueeze(1)
      T = self.forward(self.BCpoints[0], t)
      T_x = self.derivative(T, self.BCpoints[0])
      
      minusx_loss = torch.cat((minusx_loss, (T_x-self.Bi_b*T)**2), 0)
      
      T1 = self.forward(self.BCpoints[1], t)
      T_x1 = self.derivative(T1, self.BCpoints[1])
     
      x_loss = torch.cat((x_loss, (T_x1+self.Bi_t*(T1-1))**2), 0)
    
    # loss for initial condition
    
    t0_loss = torch.empty(0).to(device)
    for x in self.ICs[0]:
      x = x.unsqueeze(1)
      T = self.forward(x, self.ICs[1])
      t0_loss = torch.cat((t0_loss, (T-1)**2), 0)

    # loss for physics sample
    phys_loss = torch.empty(0).to(device)
    for i in range(len(x_points)):

      x = x_points[i].view(-1, 1)
      t = t_points[i].view(-1, 1)

      T = self.forward(x, t)
      T_x = self.derivative(T, x)
      T_xx = self.derivative(T_x, x)
      T_t = self.derivative(T, t)


      phys_loss = torch.cat((phys_loss, (T_t-4*T_xx)**2), 0)
    
    loss = torch.mean(minusx_loss) + torch.mean(x_loss) + torch.mean(t0_loss) + torch.mean(phys_loss)

    return loss
  
  def trainAdam(self, epochs_max, lr, N):

    # initialise parameters and optimiser
    self.lossTotal = self.lossTotal or [1]
    loss_val = self.lossTotal[-1]
    epoch = len(self.lossTotal)-1
    optimizer = torch.optim.Adam(self.parameters(), lr = lr)
    

    while epoch < epochs_max and loss_val > 1e-5:
      epoch += 1
      optimizer.zero_grad()
      # find loss
      loss = self.losses(N)
      loss_val = loss.item()
      # store parameter and loss values
      self.lossTotal.append(loss.item())
      # backpropagation
      loss.backward()
      optimizer.step()
      # print loss and parameter values
      print(f'Epoch: {epoch}  Loss: {round(loss_val,6)}', end='\r')
      if epoch in [1, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]: self.plot()
      
  def trainLBFGS(self, epochs_max, N):
    
    self.lossTotal = self.lossTotal or [1]
    loss_val = self.lossTotal[-1]
    epoch = len(self.lossTotal)-1
    
    optimizer = torch.optim.LBFGS(self.parameters(), max_iter=1000, history_size=1000, tolerance_grad=1.0 * np.finfo(float).eps, tolerance_change=1.0 * np.finfo(float).eps, line_search_fn='strong_wolfe')
    
    def closure():
      optimizer.zero_grad()
      loss = self.losses(N)
      loss.backward()
      return loss
  
    while epoch < epochs_max and loss_val > 1e-5:
      epoch += 1
      loss = optimizer.step(closure)
      loss_val = loss.item()
      self.lossTotal.append(loss_val)
      
      print(f'Epoch: {epoch}  Loss: {round(loss_val,6)}', end='\r')
      if epoch in [1, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]: self.plot()
      
  def plot(self):
    X = torch.linspace(-1, 1, 100).to(device)
    T = torch.linspace(0, 10, 100).to(device)

    Temp = np.zeros((len(X), len(T)))
    for i in range(len(X)):
      x = X[i].view(-1, 1)
      for j in range(len(T)):
        t = T[j].view(-1, 1)
        temp = pinn.forward(x, t)
        Temp[i, j] = temp.item()

    fig = go.Figure(data=[go.Surface(
    x=X.cpu().detach().numpy(),
    y=T.cpu().detach().numpy(),
    z=Temp,
    colorscale='magma'
    )])
    fig.update_layout(scene = dict(xaxis_title='X',yaxis_title='Time',zaxis_title='Temperature',xaxis_backgroundcolor="rgba(0, 0, 0, 0)",yaxis_backgroundcolor="rgba(0, 0, 0, 0)",zaxis_backgroundcolor="rgba(0, 0, 0, 0)"),
                      scene_camera = dict(eye=dict(x=-2, y=-2, z=2.5)),
                      plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor= 'rgba(0, 0, 0, 0)', font_color="white",
                      width=800,
                      height=800)
    fig.show()
    
    
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
# nn seed
torch.manual_seed(123)

# define nn to train, two inputs for x and t
pinn = Network(2, 6, 100, 1).to(device)

# training things
epochs_max = 1000
lrate = 1e-4
N = 100
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.cuda.amp import autocast, GradScaler
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import plotly.colors as pc
import time
import pandas as pd


class Network(nn.Module):

    def __init__(self, width: int, depth: int):
        super().__init__()

        # activation function
        activation = nn.Tanh

        # input layer
        self.input = nn.Linear(3, depth)
        init.kaiming_normal_(self.input.weight,nonlinearity='tanh')

        # hidden layers
        self.hidden = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(depth, depth), activation())
                for _ in range(width - 2)
            ]
        )

        for layer in self.hidden.children():
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight,nonlinearity='tanh')

        # output layer
        self.output = nn.Linear(depth, 1)

        self.constants()

    def constants(self) -> None:
        self.L: float = 0.1  # m
        self.sigma: float = 0.5

        self.k: int = 15  # W/mK

        self.ht: int = 200
        self.hb: int = 100

        self.Bi_t: float = self.ht * self.L / self.k
        self.Bi_b: float = self.hb * self.L / self.k

        self.lossTotal: list[float] = [1]
        self.epochData: dict[int, np.ndarray] = {}

        self.epochParams = {}

        pio.templates["dark"] = go.layout.Template(
            {
                "layout": {
                    "font": {"color": "white", "family": "Artifakt Element"},
                    "plot_bgcolor": "rgba(0,0,0,0)",
                    "paper_bgcolor": "rgba(0,0,0,0)",
                    "xaxis": {"color": "white"},
                    "yaxis": {"color": "white"},
                    "scene_zaxis": {"color": "white"},
                    "scene_camera": {"eye": {"x": 1.25, "y": 1.25, "z": 1.25}},
                }
            }
        )

        pio.templates["light"] = go.layout.Template(
            {
                "layout": {
                    "font": {"size": 14, "family": "CMU Serif"},
                    "scene": {
                        "yaxis": {
                            "gridwidth": 4,
                            "gridcolor": "rgb(242,242,242)",
                        },
                        "xaxis": {
                            "gridwidth": 4,
                            "gridcolor": "rgb(242,242,242)",
                        },
                        "zaxis": {
                            "gridwidth": 4,
                            "gridcolor": "rgb(242,242,242)",
                        },
                    },
                    "xaxis": {
                        "automargin": True,
                        "gridcolor": "rgb(242,242,242)",
                        "linecolor": "rgb(36,36,36)",
                        "showgrid": True,
                        "showline": True,
                        "ticks": "outside",
                        "title": {"standoff": 15},
                        "zeroline": False,
                    },
                    "yaxis": {
                        "automargin": True,
                        "gridcolor": "rgb(242,242,242)",
                        "linecolor": "rgb(36,36,36)",
                        "showgrid": True,
                        "showline": True,
                        "ticks": "outside",
                        "title": {"standoff": 15},
                        "zeroline": False,
                    },
                }
            }
        )
        self.setPresentationMode(False)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        inp = torch.cat((x, y, t), dim=1)
        inputted = self.input(inp)
        hidden = self.hidden(inputted)
        out = self.output(hidden)
        return out

    def init_boundaries(self, N: int) -> None:

        # Defines a vertical tensor of -sigma and sigma y values, then extends the vector to a 3D tensor for
        # boundary training then reshapes it to 4D column tensor for inputting into the network
        self.y0_boundary: torch.Tensor = (
            torch.full((N, 1), -self.sigma)
            .expand(N, N, N)
            .reshape(N**3, 1)
            .requires_grad_(True)
            .to(device)
        )
        self.y1_boundary: torch.Tensor = (
            torch.full((N, 1), self.sigma)
            .expand(N, N, N)
            .reshape(N**3, 1)
            .requires_grad_(True)
            .to(device)
        )

        # Same operations as above but for random values in the range of -sigma and sigma for initial condition
        self.y_rand: torch.Tensor = (
            (-2 * self.sigma * torch.rand(N, 1) + self.sigma)
            .expand(N, N, N)
            .reshape(N**3, 1)
            .requires_grad_(True)
            .to(device)
        )

        # Defines a vertical tensor of -1 and 1 x values, then extends the vector to a 3D tensor for
        # boundary training then reshapes it to 4D column tensor for inputting into the network
        self.x0_boundary: torch.Tensor = (
            torch.full((N,), -1.0)
            .expand(N, N, N)
            .reshape(N**3, 1)
            .requires_grad_(True)
            .to(device)
        )
        self.x1_boundary: torch.Tensor = (
            torch.full((N,), 1.0)
            .expand(N, N, N)
            .reshape(N**3, 1)
            .requires_grad_(True)
            .to(device)
        )

        # Same operations as above but for random values in the range of -1 and 1 for initial condition
        self.x_rand: torch.Tensor = (
            (-2 * torch.rand(N) + 1.0)
            .expand(N, N, N)
            .reshape(N**3, 1)
            .requires_grad_(True)
            .to(device)
        )

        # Defines a vertical tensor of 0 t values, then extends the vector to a 3D tensor for
        # initial condition training then reshapes it to 4D column tensor for inputting into the network
        self.t0_boundary: torch.Tensor = (
            torch.tensor(0.0)
            .expand(N, N, N)
            .reshape(N**3, 1)
            .requires_grad_(True)
            .to(device)
        )

        # Same operations as above but for random values in the range of 0 and 10 for boundary training
        self.t_rand: torch.Tensor = (
            (10*torch.rand(N, 1, 1))
            .expand(N, N, N)
            .reshape(N**3, 1)
            .requires_grad_(True)
            .to(device)
        )

    def derivative(self, f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(
            f, x, grad_outputs=torch.ones_like(f).to(device), create_graph=True
        )[0]

    def losses(self, N: int) -> torch.Tensor | None:

        self.init_boundaries(N)

        # -1x all y values boundary condition is cooling heat transfer in the y direction
        T: torch.Tensor = self.forward(self.x0_boundary, self.y_rand, self.t_rand)
        T_x0: torch.Tensor = self.derivative(T, self.x0_boundary)

        minusx_loss: torch.Tensor = torch.mean((T_x0 - self.Bi_b * T) ** 2)

        # 1x all y values boundary condition is heating heat transfer in the y direction
        T2: torch.Tensor = self.forward(self.x1_boundary, self.y_rand, self.t_rand)
        T_x1: torch.Tensor = self.derivative(T2, self.x1_boundary)

        x_loss: torch.Tensor = torch.mean(
            (T_x1 + self.Bi_t * (T2 - 1)) ** 2
        )

        # -sigma y all x values boundary condition is adiabatic in the x direction
        T3: torch.Tensor = self.forward(self.x_rand, self.y0_boundary, self.t_rand)
        T_y0: torch.Tensor = self.derivative(T3, self.y0_boundary)

        minusy_loss: torch.Tensor = torch.mean(T_y0**2)

        # sigma y all x values boundary condition is adiabatic in the x direction
        T4: torch.Tensor = self.forward(self.x_rand, self.y1_boundary, self.t_rand)
        T_y1: torch.Tensor = self.derivative(T4, self.y1_boundary)

        y_loss: torch.Tensor = torch.mean(T_y1**2)

        # loss for initial condition

        T_IC: torch.Tensor = self.forward(self.x_rand, self.y_rand, self.t0_boundary)
        t0_loss: torch.Tensor = torch.mean((T_IC - 1) ** 2)

        # loss for physics sample
        T_phy: torch.Tensor = self.forward(self.x_rand, self.y_rand, self.t_rand)
        
        T_x_phy: torch.Tensor = self.derivative(T_phy, self.x_rand)
        T_xx_phy: torch.Tensor = self.derivative(T_x_phy, self.x_rand)
        
        T_y_phy: torch.Tensor = self.derivative(T_phy, self.y_rand)
        T_yy_phy: torch.Tensor = self.derivative(T_y_phy, self.y_rand)
        
        T_t_phy: torch.Tensor = self.derivative(T_phy, self.t_rand)

        phys_loss: torch.Tensor = torch.mean(
            (T_t_phy - 4 * (T_xx_phy + (T_yy_phy / (self.sigma**2) )) ) ** 2
        )

        loss: torch.Tensor = (
            minusx_loss + x_loss + minusy_loss + y_loss + t0_loss + phys_loss
        )
        return loss

    def train(
        self, epochs_max: int, N: int, optimiser: str = "LBFGS", lr: float = 1
    ) -> None:

        def closure() -> float:
            optimizer.zero_grad()
            loss: torch.Tensor = self.losses(N)
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.parameters(), clip_value=1.0)
            return loss

        loss_val: float = self.lossTotal[-1]
        epoch: int = len(self.lossTotal) - 1
        epoch_times: list[float] = []
        window_size: int = 10

        scaler = GradScaler()

        if optimiser == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimiser == "LBFGS":
            optimizer = torch.optim.LBFGS(
                self.parameters(),
                max_iter=1000,
                history_size=1000,
                line_search_fn="strong_wolfe",
            )
        else:
            raise ValueError("Invalid optimiser")

        start: float = time.time()

        while epoch < epochs_max and loss_val > 1e-6:
            epoch += 1
            if optimiser == "Adam":
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    loss: torch.Tensor = self.losses(N)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                torch.nn.utils.clip_grad_value_(
                    self.parameters(), clip_value=1.0
                )  # clip the unscaled gradients
                scaler.step(optimizer)
                scaler.update()
            else:
                loss: torch.Tensor = optimizer.step(closure)

            loss_val: float = loss.item()
            self.lossTotal.append(loss_val)

            # compute epoch time
            epoch_time: float = time.time() - start
            epoch_times.append(epoch_time)
            start: float = time.time()  # Reset the start time for the next epoch

            # Compute the moving average of epoch times
            if len(epoch_times) > window_size:
                epoch_times: list[float] = epoch_times[-window_size:]
            avg_epoch_time: float = sum(epoch_times) / len(epoch_times)

            # Calculate the remaining time
            time_remaining: float = (epochs_max - epoch) * avg_epoch_time
            self.epochParams[epoch] = [
                torch.mean(i).cpu().detach().numpy() for i in self.parameters()
            ]

            # print loss and parameter values
            print(
                f'Epoch: {epoch}  Loss: {loss_val:.8f} Time/Epoch: {avg_epoch_time:.4f}s ETA: {time.strftime("%H Hours %M Minutes and %S Seconds", time.gmtime(time_remaining))}',
                end="\r",
            )
            if epoch in (
                [1, 50, 100, 200, 500, 1000, 2000, 5000]
                + list(range(5000, 100001, 10000))
            ):
                self.storeData(epoch)

        self.storeData(epoch)

    def storeData(self, epoch: int) -> None:
        X, Y, T = torch.meshgrid(
            torch.linspace(-1, 1, 50),
            torch.linspace(-self.sigma, self.sigma, 50),
            torch.linspace(0, 10, 50),
        )

        Temp: np.ndarray = (
            (
                pinn.forward(
                    X.flatten().unsqueeze(1).to(device),
                    Y.flatten().unsqueeze(1).to(device),
                    T.flatten().unsqueeze(1).to(device),
                )
            )
            .reshape(50, 50, 50)
            .cpu()
            .detach()
            .numpy()
        )  # Predicted temperature
        self.epochData[epoch] = Temp

    def setPresentationMode(self, state: bool) -> None:
        if state:
            self.theme: str = "light"
            pio.renderers.default = "png"
        else:
            self.theme: str = "dark"
            pio.renderers.default = "plotly_mimetype+notebook"

    def plotEpochs(self) -> None:
        # Create figure
        fig: go.Figure = go.Figure()

        fig.update_layout(
            template=self.theme,
            title=dict(text="<b>Temperature Distribution</b>", font=dict(size=30)),
            scene=dict(
                xaxis=dict(
                    title="X",
                    showgrid=True,
                    zeroline=True,
                    zerolinewidth=5,
                    zerolinecolor="white",
                ),
                yaxis=dict(
                    title="Y",
                    showgrid=True,
                    zeroline=True,
                    zerolinewidth=5,
                    zerolinecolor="white",
                ),
                zaxis=dict(
                    title="Time (s)",
                    showgrid=False,
                    zeroline=True,
                    zerolinewidth=5,
                    zerolinecolor="white",
                    range=[0, 10],
                ),
            ),
            scene_camera=dict(
                eye=dict(x=-2, y=1, z=1.5),
                up=dict(x=0, y=1, z=0),
            ),
            width=800,
            height=800,
        )

        X: np.ndarray = np.linspace(-1, 1, 50)
        Y: np.ndarray = np.linspace(-self.sigma, self.sigma, 50)
        T: np.ndarray = np.linspace(0, 10, 50)

        if self.theme == "dark":
            # Add traces, one for each slider step
            for trace in self.epochData.keys():
                fig.add_trace(
                    go.Scatter3d(
                        visible=False,
                        x=X,
                        y=Y,
                        z=T,
                        mode = "markers",
                        marker=dict(
                            size=4,  # Adjust marker size as needed
                            color=self.epochData[trace].flatten(),  # Map temperature to color
                            colorscale='Agsunset',  # Choose a colorscale
                        )
                    )
                )

            # Make first trace visible
            fig.data[-1].visible = True

            # Create and add slider
            steps = []
            for i, epoch in enumerate(self.epochData.keys()):
                step = dict(
                    method="update",
                    args=[
                        {"visible": [False] * len(fig.data)},
                    ],
                    label=str(epoch),
                )
                step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
                steps.append(step)

            sliders = [
                dict(
                    active=len(self.epochData) - 1,
                    currentvalue={"prefix": "Epoch: "},
                    pad={"t": 5},
                    steps=steps,
                )
            ]
            fig.update_layout(sliders=sliders)
        else:
            last_epoch: int = list(self.epochData.keys())[-1]
            fig.add_trace(
                go.Scatter3d(
                    x=X,
                    y=Y,
                    z=T,
                    mode = "markers",
                    marker=dict(
                        size=4,  # Adjust marker size as needed
                        color=self.epochData[last_epoch].flatten(),  # Map temperature to color
                        colorscale='Agsunset',  # Choose a colorscale
                    )
                )
            )

        fig.show()

    def plotIC(self) -> None:
        fig: go.Figure = go.Figure()

        fig.update_layout(
            template=self.theme,
            title=dict(text="<b>Initial Condition</b>", font=dict(size=30)),
            xaxis_title="<b>X</b>",
            yaxis_title="<b>Y</b>",
            legend_title="<b>Epochs</b>",
            height=800,
            width=1600,
        )

        if self.theme != "dark":
            last_epoch: int = list(self.epochData.keys())[-1]
            fig.add_trace(
                go.Scatter(
                    x=np.linspace(-1, 1, 100),
                    y=self.epochData[last_epoch][:, 0],
                    line={"color": "black"},
                )
            )
            fig.show()
            return

        ICdata: dict[int, np.ndarray] = {}
        for key, matrix in self.epochData.items():
            ICdata[key] = matrix[:, 0]  # Get the first column of each matrix

        del ICdata[1]  # remove the first epoch, as it's erroneous

        lines: int = len(ICdata)

        colours: list[str] = pc.sample_colorscale(
            "Agsunset", lines, low=0.0, high=1.0, colortype="rgb"
        )  # Get colours for each epoch

        df: pd.DataFrame = pd.DataFrame(ICdata)
        df.index = np.linspace(-1, 1, 100)

        for i, epoch in enumerate(ICdata.keys()):
            fig.add_trace(
                go.Scatter(
                    visible=True,
                    x=df.index,
                    y=df[epoch],
                    line={"color": colours[i]},
                    name=f"Epoch {epoch}",
                )
            )  # Add traces

        # Create and add slider
        steps = [
            dict(
                method="update",
                args=[
                    {"visible": [True] * lines},
                ],
                label="Default view",
            )
        ]  # Default view
        for i in range(lines - 1):
            steps.append(
                dict(
                    method="update",
                    args=[
                        {"visible": [False] * (i + 1) + [True] * (lines - i - 1)},
                    ],
                    label=f"More than {df.columns[i]} epochs",
                )
            )  # Removes visibility of consecutive epochs with each step

        # Add slider
        sliders = [dict(active=0, pad={"t": 50}, steps=steps)]

        fig.update_layout(sliders=sliders)

        fig.show()

    def plotLoss(self) -> None:
        fig: px.Figure = px.scatter(
            x=range(len(self.lossTotal)),
            y=self.lossTotal,
            log_y=True,
            log_x=True,
            color=np.log(self.lossTotal),
            color_continuous_scale="Agsunset",
            labels={"x": "<b>Epochs (log)</b>", "y": "<b>Loss (log)</b>"},
            trendline="lowess",
            trendline_options=dict(frac=0.005),
            trendline_color_override="midnightblue",
            template=self.theme,
        )
        fig.update_layout(
            title=dict(
                text="<b>Loss against Epoch</b>",
                font=dict(size=30),
                automargin=True,
                pad={"t": 25},
            ),
            coloraxis_showscale=False,
            height=800,
            width=1600,
        )
        fig.update_traces(marker=dict(size=6, opacity=0.8))

        fig.show()
        
        
        
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# nn seed
torch.manual_seed(987)

# define nn to train
pinn = Network(6, 50).to(device)

# training things
epochs_max = 2000
lrate = 1e-4
N = 50

pinn.train(epochs_max, N, 'Adam', lrate)
pinn.train(epochs_max*2,N)


pinn.plotLoss()


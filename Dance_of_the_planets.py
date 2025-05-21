import numpy as np
import torch
from scipy.integrate import solve_ivp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

G = 6.67430e-11

# Input parameters
m1 = 1.0e24
m2 = 1.0e24
x1_0 = np.array([0.0, 0.0])
x2_0 = np.array([1.0e6, 0.0])
v1_0 = np.array([0.0, 0.0])
v2_0 = np.array([0.0, 1.0e3])
t_pred = 3.6e3
num_epochs = 100

# Trajectory generator
def generate_trajectory(m1, m2, x1_0, x2_0, v1_0, v2_0, t_span, n_points=100):
    def derivatives(t, y):
        x1, x2 = y[:2], y[2:4]
        r = x2 - x1
        r_mag = np.linalg.norm(r) + 1e-6
        a1 = G * m2 * r / r_mag**3
        a2 = -G * m1 * r / r_mag**3
        return np.concatenate([y[4:6], y[6:8], a1, a2])

    sol = solve_ivp(derivatives, [0, t_span], np.concatenate([x1_0, x2_0, v1_0, v2_0]),
                    t_eval=np.linspace(0, t_span, n_points), rtol=1e-9, atol=1e-9)
    return sol.t, sol.y.T

# Data generator
def generate_training_data(num_samples=100, t_span=3600):
    data = []
    for _ in range(num_samples):
        m1 = 10**np.random.uniform(23, 25)
        m2 = 10**np.random.uniform(23, 25)
        x1_0 = np.random.uniform(-1e6, 1e6, 2)
        delta = np.random.uniform(1e5, 1e6)
        angle = np.random.uniform(0, 2*np.pi)
        x2_0 = x1_0 + delta * np.array([np.cos(angle), np.sin(angle)])

        r = x2_0 - x1_0
        r_mag = np.linalg.norm(r)
        orbital_speed = np.sqrt(G * (m1 + m2) / r_mag)
        direction = np.array([-np.sin(angle), np.cos(angle)])
        v1_0 = orbital_speed * m2 / (m1 + m2) * direction
        v2_0 = -orbital_speed * m1 / (m1 + m2) * direction

        t, states = generate_trajectory(m1, m2, x1_0, x2_0, v1_0, v2_0, t_span)
        for time, state in zip(t, states):
            data.append({
                'm1': m1, 'm2': m2, 'x1_0': x1_0, 'x2_0': x2_0,
                'v1_0': v1_0, 'v2_0': v2_0, 't': time,
                'x1_t': state[:2], 'x2_t': state[2:4]
            })
    return data

train_data = generate_training_data(500)
test_data = generate_training_data(100)

# Model
class TwoBodyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 4)
        )

    def forward(self, inputs):
        return self.net(inputs)

model = TwoBodyNet()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

# Dataloader

def collate_fn(batch):
    inputs, targets = [], []
    for sample in batch:
        m1 = sample['m1'] / 1e24
        m2 = sample['m2'] / 1e24
        x1_0 = sample['x1_0'] / 1e6
        x2_0 = sample['x2_0'] / 1e6
        v1_0 = sample['v1_0'] / 1e3
        v2_0 = sample['v2_0'] / 1e3
        t = sample['t'] / 3600

        input_vec = torch.tensor([t, m1, m2, *x1_0, *x2_0, *v1_0, *v2_0], dtype=torch.float32)

        x1_t = sample['x1_t'] / 1e6
        x2_t = sample['x2_t'] / 1e6
        target_vec = torch.tensor([*(x1_t - x1_0), *(x2_t - x2_0)], dtype=torch.float32)

        inputs.append(input_vec)
        targets.append(target_vec)

    return torch.stack(inputs), torch.stack(targets)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=32, collate_fn=collate_fn)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.functional.mse_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        #print(f"Batch Loss: {loss.item():.4e}")

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            val_loss += nn.functional.mse_loss(outputs, targets).item()

    avg_train = total_loss / len(train_loader)
    avg_val = val_loss / len(test_loader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_train:.4e} | Val Loss: {avg_val:.4e}")
    scheduler.step(avg_val)

from mpl_toolkits.mplot3d import Axes3D

# Evaluation with nudge error
t, states = generate_trajectory(m1, m2, x1_0, x2_0, v1_0, v2_0, t_pred)
true_x1 = states[:, :2]
true_x2 = states[:, 2:4]

t_eval = np.linspace(0, t_pred, 100)
preds = []
model.eval()
with torch.no_grad():
    for time in t_eval:
        inp = torch.tensor([
            time / 3600, m1/1e24, m2/1e24,
            *(x1_0/1e6), *(x2_0/1e6),
            *(v1_0/1e3), *(v2_0/1e3)
        ], dtype=torch.float32)
        
        inp += torch.randn_like(inp) * 0.01  # Small noise

        delta_pos = model(inp)
        pred = delta_pos.numpy().reshape(2, 2) + np.stack([x1_0, x2_0])/1e6
        preds.append(pred)

preds = np.array(preds) * 1e6
pred_x1 = preds[:, 0]
pred_x2 = preds[:, 1]

# Compute MAE
mae_x1 = np.mean(np.abs(pred_x1 - true_x1))
mae_x2 = np.mean(np.abs(pred_x2 - true_x2))
print(f"MAE Body 1: {mae_x1:.2f} m")
print(f"MAE Body 2: {mae_x2:.2f} m")

# 3D Plot with Time as Z
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

time_hours = t_eval / 3600

# True Trajectories
ax.plot(true_x1[:, 0], true_x1[:, 1], time_hours, 'b-', label='True Body 1')
ax.plot(true_x2[:, 0], true_x2[:, 1], time_hours, 'g-', label='True Body 2')

# Predicted Trajectories
ax.plot(pred_x1[:, 0], pred_x1[:, 1], time_hours, 'r--', label='Pred Body 1')
ax.plot(pred_x2[:, 0], pred_x2[:, 1], time_hours, 'm--', label='Pred Body 2')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Time (hours)')
ax.set_title('2D Orbits in 3D Plot with Time')
ax.legend()
plt.tight_layout()
plt.show()

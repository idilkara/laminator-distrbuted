import torch
import torch.nn as nn
import torch.nn.functional as F
import zmq
import time, random

TASK_ENDPOINT = "tcp://coordinator:5557"
RESULT_ENDPOINT = "tcp://coordinator:5558"

# A simple NN 
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

def compute_grad_and_loss(task):
    X = torch.tensor(task["X"], dtype=torch.float32)
    y = torch.tensor(task["y"], dtype=torch.float32).unsqueeze(1)
    w_state = task["weights"]

    # Build model & load weights
    model = Net(input_dim=X.shape[1])
    model.load_state_dict(w_state)

    # Loss
    criterion = nn.BCELoss()

    # Forward
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # Backward
    model.zero_grad()
    loss.backward()

    # Collect gradients
    grads = {name: p.grad.numpy() for name, p in model.named_parameters()}
    return grads, float(loss.item()), X.shape[0]

def main():
    ctx = zmq.Context()
    receiver = ctx.socket(zmq.PULL); receiver.connect(TASK_ENDPOINT)
    sender = ctx.socket(zmq.PUSH); sender.connect(RESULT_ENDPOINT)

    time.sleep(1)

    while True:
        task = receiver.recv_pyobj()
        grads, loss, n = compute_grad_and_loss(task)
        time.sleep(random.uniform(0.05, 0.2))
        sender.send_pyobj({
            "worker_id": task["worker_id"],
            "grads": grads,
            "loss": loss,
            "n": n,
        })

if __name__ == "__main__":
    main()

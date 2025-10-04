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

from models import LinearNet

def compute_grad_and_loss(task):
    X = torch.tensor(task["X"], dtype=torch.float32)
    y = torch.tensor(task["y"], dtype=torch.long)  # classification labels
    state_dict = task["weights"]

    model = LinearNet([128, 256, 128])
    model.load_state_dict(state_dict)
    model.train()

    criterion = nn.CrossEntropyLoss()
    outputs = model(X)
    loss = criterion(outputs, y)

    loss.backward()

    grads = {name: p.grad.clone().numpy() for name, p in model.named_parameters()}
    return grads, float(loss.item()), X.shape[0]

def main():
    ctx = zmq.Context()
    receiver = ctx.socket(zmq.PULL); receiver.connect(TASK_ENDPOINT)
    sender = ctx.socket(zmq.PUSH); sender.connect(RESULT_ENDPOINT)

    time.sleep(1)

    while True:
        task = receiver.recv_pyobj()
        print("Received task from the coordinator",flush=True)
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

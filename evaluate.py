from model.position_net import PositionNet

model = PositionNet()
model.load_state_dict(torch.load("PositionNet.pt"))

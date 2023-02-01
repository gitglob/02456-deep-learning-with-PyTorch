import matplotlib.pyplot as plt
import time
from PIL import Image

import gym

import torch
import torchvision.transforms as T

from functions import *
from utils import *
from classes import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f">> Using device: {device}")

game = "pacman"

def test_network():
    # test the network
    model = DQN(9, 0.001)
    model = model.float()
    model.to(device)

    # create fake input
    x = torch.zeros(64, 210, 160, 3)
    x_img = Image.new('RGB', (210, 160))
    x_img_1 = resize_image(x_img, 160)

    # p = T.Compose([
    #     T.Resize((160,160))
    # ])
    # x_img_2 = p(x_img)
    # print(np.shape(x_img_2))

    # convert to tensor and reshape appropriately
    transforms = T.Compose([
        T.ToTensor()
    ])
    input = transforms(x_img_1)
    input = input.reshape(1, 160, 160, 3)
    input = input.permute(0, 3, 1, 2)
    input = input.to(device).float()
    print("Input shape: ", input.shape)
    print("Input type: ", type(input))

    # do one forward pass and check output shape
    model.train()
    y = model(input)
    print(y.shape)

def pacman(game):
    # hyperparameters
    num_eps = 3000 # number of episodes
    ep_lim = 100 # episode limit
    batch_size = 64
    lr = 0.005 # learning rate
    dr = 0.99 # discount rate
    tau = 0.01 # target network update rate
    vf = 100 # validation frequency
    eps_start = 1 # starting epsilon

    # create environment
    env = gym.make(game)

    # define input and output
    input_shape = (210, 160, 3)
    n_outputs = env.action_space.n

    # initialize networks
    replay_memory_capacity = 10000
    prefill_memory = True

    # initialize DQN and replay memory
    policy_net = DQN(n_outputs, lr).float().to(device)
    target_net = DQN(n_outputs, lr).float().to(device)
    target_net.load_state_dict(policy_net.state_dict())

    replay_mem = ReplayMemory(replay_memory_capacity)

    # prefill replay memory with random actions
    if replay_mem:
        print('prefill replay memory')
        initial_state = env.reset()
        # initial_state[0]: numpy.ndarray - (210, 160, 3)
        # initial_state[1]: {'lives': 3, 'episode_frame_number': 0, 'frame_number': 29997}
        s = initial_state[0]
        while replay_mem.count() < replay_memory_capacity:
            a = env.action_space.sample()
            s1, r, terminated, truncated, _ = env.step(a)
            # s1: numpy.ndarray - (210, 160, 3)
            done = terminated or truncated
            replay_mem.add(s, a, r, s1, done)
            if not done:
                s = s1
            else:
                initial_state = env.reset()
                s = initial_state[0]

    # train network
    rewards, lengths, losses, epsilons = train(num_eps, ep_lim, lr, dr, vf, eps_start, 
                                tau, batch_size, 
                                replay_mem, 
                                policy_net, target_net,
                                env,
                                device)


    # plot results
    plt.figure(figsize=(16, 9))
    plt.subplot(411)
    plt.title('training rewards')
    plt.plot(range(1, num_eps+1), rewards)
    plt.plot(moving_average(rewards))
    plt.xlim([0, num_eps])
    plt.subplot(412)
    plt.title('training lengths')
    plt.plot(range(1, num_eps+1), lengths)
    plt.plot(range(1, num_eps+1), moving_average(lengths))
    plt.xlim([0, num_eps])
    plt.subplot(413)
    plt.title('training loss')
    plt.plot(range(1, num_eps+1), losses)
    plt.plot(range(1, num_eps+1), moving_average(losses))
    plt.xlim([0, num_eps])
    plt.subplot(414)
    plt.title('epsilon')
    plt.plot(range(1, num_eps+1), epsilons)
    plt.xlim([0, num_eps])
    plt.tight_layout(); plt.show()

def demo(game):
    print("Executing demo...")
    env = gym.make(game)
    env.reset()
    for i in range(50):
        # env.render()
        s, r, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        if i == 0:
            print("Step output: ", s.shape, r, done, i)
    
    env.close()

def main():
    game = "MsPacman-v4"
    # demo(game)
    # test_network()
    pacman(game)
    return 

if __name__ == "__main__":
    main()
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from utils import resize_image

P = T.Compose([
    T.Resize((60, 60))
])

def prepare_img(img_list):
    img = np.array(img_list, dtype=np.uint8)
    img = Image.fromarray(img)
    img = P(img)
    # (60, 60, 3)

    return img

def prepare(input, batch_size):
    output = []
    x_arr = np.array(input, dtype=np.uint8)
    for x in input:
        img = prepare_img(x)
        # (160, 160, 3)
        img = np.array(img).tolist()
        output.append(img)
        
    output_arr = np.array(output)
    output_tensor = torch.from_numpy(output_arr)
    # (batch_size, 160, 160, 3)

    return output_tensor

# training loop
def train(num_episodes, episode_limit, learning_rate, gamma, val_freq, epsilon_start, 
          tau, batch_size, 
          replay_memory, 
          policy_dqn, target_dqn,
          env,
          device):   
    try:
        print('start training')
        epsilon = epsilon_start
        rewards, lengths, losses, epsilons = [], [], [], []
        for i in range(num_episodes):
            print(f"Episode: {i}")
            # init new episode
            initial_state, ep_reward, ep_loss = env.reset(), 0, 0
            # initial_state[0]: numpy.ndarray - (210, 160, 3)
            # initial_state[1]: {'lives': 3, 'episode_frame_number': 0, 'frame_number': 29997}
            s = initial_state[0]
            # print("Resetting environment...")
            print("Iterating over episode limit...")
            for j in range(episode_limit):
                print(f"Sub-episode: {j}")
                # select action with epsilon-greedy strategy
                if np.random.rand() < epsilon:
#                     print("Exploring...")
                    a = env.action_space.sample()
                else:
#                     print("Exploiting...")
                    with torch.no_grad():
                        s_in = np.array(prepare_img(s))
                        s_in = torch.from_numpy(s_in).float().to(device)
                        a = policy_dqn(s_in)
                        a = a.argmax().item()
                # perform action
                # print("Performing action...")
                s1, r, terminated, truncated, _ = env.step(a)
                # s1: numpy.ndarray - (210, 160, 3)
                done = terminated or truncated
                # store experience in replay memory
                replay_memory.add(s, a, r, s1, done)
                # batch update
                if replay_memory.count() >= batch_size:
                    # sample batch from replay memory
                    batch = replay_memory.sample(batch_size)
                    # print("Sampling batch from memory...")
                    ss, aa, rr, ss1, dd = [], [], [], [], []
                    for b in batch:
                        s, a, r, s1, d = b
                        ss.append(s)
                        aa.append(a)
                        rr.append(r)
                        ss1.append(s1)
                        dd.append(d)
                    # print("Converted to batch lists...")
                    # (64, 210, 160, 3)
                    ss = prepare(ss, batch_size).float().to(device)
                    ss1 = prepare(ss1, batch_size).float().to(device)
                    # (64, 60, 60, 3)
                    # do forward pass of batch
                    policy_dqn.optimizer.zero_grad()
                    # print("Forward pass...")
                    Q = policy_dqn(ss)
                    # use target network to compute target Q-values
                    with torch.no_grad():
                        # use target net
                        # print("Compute target values...")
                        Q1 = target_dqn(ss)
                    # compute target for each sampled experience
                    q_targets = Q.clone()
                    for k in range(batch_size):
                        q_targets[k, aa[k]] = rr[k] + gamma * Q1[k].max().item() * (not dd[k])
                    # update network weights
                    loss = policy_dqn.loss(Q, q_targets).to(device)
                    loss.backward()
                    policy_dqn.optimizer.step()
                    # update target network parameters from policy network parameters
                    target_dqn.update_params(policy_dqn.state_dict(), tau)
                else:
                    loss = 0
                # bookkeeping
                s = s1
                ep_reward += r
                ep_loss += loss.item()
                if done: break
            
            # bookkeeping
            epsilon *= num_episodes/(i/(num_episodes/20)+num_episodes) # decrease epsilon
            epsilons.append(epsilon); rewards.append(ep_reward); lengths.append(j+1); losses.append(ep_loss)
            
            if (i+1) % val_freq == 0: 
                print('%5d mean training reward: %5.2f' % (i+1, np.mean(rewards[-val_freq:])))
        
        print('done')
    except KeyboardInterrupt:
        print('interrupt')

    return rewards, lengths, losses, epsilons


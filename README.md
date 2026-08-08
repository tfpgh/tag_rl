# Tag, You're It

Two custom robots learn to play tag in simulation through self-play reinforcement learning, then transfer to the real world with no fine-tuning.

**[Watch them play](https://youtu.be/83YcsIVBEEk)** · **[Poster](https://tobypenner.com/tag/poster.pdf)**

[Gameplay](https://github.com/user-attachments/assets/fdf1c895-c2dc-47ee-b02f-4c47710d9e93)

## Motivation

I wanted to watch robots teach themselves to play tag. It’s a children’s game, how hard could it be? There are two main difficulties when applying reinforcement learning to tag:
1. **Multi-agent self-play** violates a standard RL assumption that the environment is stable. If the chaser is improving but the evader is improving faster, the chaser feels like it’s getting worse.
2. **RL is sample-inefficient**. Training takes tens of millions of games, which is infeasible on real robots. So, I
built an efficient simulation, trained in it, and transferred the policies to real robots (sim-to-real transfer).

## Approach

Each robot observes 128 rays cast in all directions, returning the distance to the first hit and whether it hit the other robot or a wall/obstacle. The actor model is a 1D CNN into a GRU into a dense action head outputting left and right motor voltage distributions.

The agents receive the following rewards in training:
|        | Per Step | Tag  | Timeout | Collision |
|--------|----------|------|---------|-----------|
| Chaser | -0.01    | +1.0 | -1.0    | -10.0     |
| Evader | +0.01    | -1.0 | +1.0    | -10.0     |

Both policies train independently under PPO across 65,536 parallel MuJoCo environments. This setup achieves three billion simulated steps in three hours on eight RTX PRO 6000 Blackwell GPUs, or about 4.75 years of tag.

It's impossible to make the simulation perfectly accurate to reality, so I randomize various environment parameters throughout training: mass, friction, motor strength, latency, sensor noise. This is the main reason I used a recurrent policy. If the policy can't maintain state across timesteps, it has to learn a general policy which works in all environment conditions. If it's recurrent, it can maintain an implicit belief about the dynamics regime it's operating in, and optimize for that.

I also manually recorded some driving on the real hardware setup, and used CMA-ES to fit the domain randomization parameters to the recorded data. This centers the domain randomization distributions on reality.

The robots themselves are very simple; they don't know anything about the game being played and they don't have any motor encoders or IMUs. They just listen to motor voltage commands over UDP at 20 Hz.

Tracking uses an overhead USB camera and AprilTags on the robots and obstacles. A desktop computer handles AprilTag tracking, runs inference, and commands the robots.

## Results

The tag rate in simulation settles near 90%, oscillating by a few percent as the two policies co-adapt, with a mean game length of 11 seconds and roughly one wall or obstacle collision per seven minutes of play.

If there are no obstacles, the chaser can easily drive back and forth pushing the evader closer to a wall until they have nowhere to go. With obstacles, the evader usually loops around one of them which is hard for the chaser to break. The evader in this scenario will often try to reverse direction while it's out of sight of the chaser.

The robots often weave excessively at the start of games for seemingly no reason. My best guess is that this is the quickest way to build up a strong implicit understanding of the environment dynamics. They don't weave if they're trained without domain randomization, which lines up with this hypothesis.

The largest room for improvement is the simulation environment. Even after system identification, simulated and real trajectories diverge in under a second. The robots have tiny wheel contact patches, so their dynamics are very unpredictable. Building a more stable robot and using a learned dynamics model would be interesting improvements to try.

---

An independent study at Middlebury College, Spring 2026, advised by [Michael Linderman](https://www.cs.middlebury.edu/~mlinderman/).

import gym
import gym_px4_offb
import time
import os
import rospy
import argparse


def timer(t):
    for _ in range(t):
        print('.', end='', flush=True)
        time.sleep(1)
    print('\n')

parser = argparse.ArgumentParser()
parser.add_argument("--use_px4_ctrl", type=bool, default=True)
parser.add_argument("--takeoff_height", type=float, default=1.0)
parser.add_argument("--px4_ctrl_mode", type=str, default="v")

args = parser.parse_args()


if __name__ == "__main__":
    env = gym.make("px4-offb-single-v0", use_px4_ctrl=args.use_px4_ctrl, takeoff_height=args.takeoff_height, px4_ctrl_mode=args.px4_ctrl_mode)

    t0 = time.time()

    for _ in range(2):
        obs = env.reset()
        for i in range(500):
            # a = env.policy()
            if i < 50:
                a = (1, 0, 0)
            elif i < 100:
                a = (0, 1, 0)
            elif i < 150:
                a = (-1, 0, 0)
            elif i < 200:
                a = (0, -1, 0)
            obs_, r, done, info = env.step(a)
            print("step-{} pos: ({:.3f}, {:.3f}, {:.3f})".format(i, obs_[1][0], obs_[1][1], obs_[1][2]))
            obs = obs_
            if "TimeLimit.truncated" in info:
                break
            if i > 200:
                env.step((0, 0, 0))
                time.sleep(2)
                break

    t1 = time.time()

    print(f"time elapse: {t1 - t0} s")

    env.close()

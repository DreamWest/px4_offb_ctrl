from gym.envs.registration import register

register(
    id='px4-offb-single-v0',
    entry_point='gym_px4_offb.envs:PX4OffbSingle',
    max_episode_steps=400
)
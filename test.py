from gym import envs
for env in envs.registry.all():
    print(env.id)
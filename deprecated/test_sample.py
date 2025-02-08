import jax


rng=jax.random.PRNGKey(1)

shape=(4,1)
x1=jax.random.randint(rng,shape,0,10)
print(x1)
def sample(rng):
    return jax.random.randint(rng,shape[1],0,10)

x2=jax.vmap(sample,)(jax.random.split(rng,4))
print(x2)

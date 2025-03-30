from invoke import task
import os

@task
def python(ctx):
    """ """
    ctx.run("which python" if os.name != "nt" else "where python")
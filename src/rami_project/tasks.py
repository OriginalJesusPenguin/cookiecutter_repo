from invoke import task
import os

@task
def python(ctx):
    """ """
    ctx.run("which python" if os.name != "nt" else "where python")

@task
def git(ctx, message):
    print(f"Running git with message: {message}")
    ctx.run(f"git add .")
    ctx.run(f"git commit -m '{message}'")
    ctx.run(f"git push")

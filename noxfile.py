import nox

@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def lint(session):
    session.install("-e", ".")
    session.install("pylint")
    session.run("pylint", "src/google", "--rcfile=pylintrc")

@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def unit(session):
    session.install("-e", ".[test]")
    session.run("pytest", "tests/unittests")

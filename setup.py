from skbuild import setup

setup(
    name="pomdp_spaceship_env",
    version="0.0.1",
    description="A Partially Observable Space Ship Environment for RL",
    author='Finn Lukas Busch',
    author_email = 'finn.lukas.busch@gmail.com'
    license="MIT",
    packages=['pomdp_spaceship_env'],
    package_dir={'': 'src'},
    cmake_install_dir='src/pomdp_spaceship_env',
    python_requires='>=3.9',
    include_package_data=True,
)

from skbuild import setup

setup(
    name="pomdp_spaceship_env",
    version="1.2.3",
    description="a minimal example package (with pybind11)",
    author='Pablo Hernandez-Cerdan',
    license="MIT",
    packages=['pomdp_spaceship_env'],
    package_dir={'': 'src'},
    cmake_install_dir='src/pomdp_spaceship_env',
    python_requires='>=3.7',
    include_package_data=True,
)

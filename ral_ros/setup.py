# ament_python package. RAL itself is a pip package: install it into the Python that ROS 2
# uses (pip install "ral-audio[dsp]"), then build this package with colcon.
from glob import glob

from setuptools import setup

package_name = "ral_ros"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="The RAL Authors",
    description="RAL as a ROS 2 node",
    license="Apache-2.0",
    entry_points={"console_scripts": ["ral_node = ral_ros.node:main"]},
)

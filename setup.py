from setuptools import setup
import os
from glob import glob
package_name = 'joystick_control_ras'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
       #('share/' + package_name + '/' + package_name, ['raslogo1.png']),
        (os.path.join('share', package_name, 'resource/images'), glob('resource/images/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Bart Boogmans',
    maintainer_email='bartboogmans@hotmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'joystickgui = joystick_control_ras.broadcastGUI:main'
        ],
    },
)

from setuptools import find_packages, setup

package_name = 'rknn_yolo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/demo.launch.py'])
    ],
    install_requires=['setuptools','pyserial'],
    zip_safe=True,
    maintainer='orangepi',
    maintainer_email='orangepi@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'crossroad_detector = rknn_yolo.crossroad_detector:main',
            'opencv_pid_1 = rknn_yolo.opencv_pid_1:main',
            'rknn_yolo = rknn_yolo.main:main',
            'camre = rknn_yolo.camre:main',
            'opencv_pid = rknn_yolo.opencv_pid:main',
            'motor = rknn_yolo.motor:main',
            'targer = rknn_yolo.targer:main'
        ],
    },
)

from setuptools import setup, find_packages

setup(
    name='MegaPortraits',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'torch==1.8.1',
        'torchvision==0.9.1',
        'Pillow==8.2.0',
        'numpy==1.20.2',
        'opencv-python==4.5.1.48',
        'matplotlib==3.4.1',
        'pyyaml==5.4.1'
    ],
    entry_points={
        'console_scripts': [
            'train=scripts.train:main',
            'train_highres=scripts.train_highres_model:main',
            'train_student=scripts.train_student_model:main',
            'inference=scripts.inference:main',
            'evaluate=scripts.evaluate:main'
        ],
    },
    python_requires='>=3.6',
)

from setuptools import setup, find_packages
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='IQA_pytorch',
    version='0.1',
    description='IQA models in PyTorch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['IQA_pytorch'],
    data_files= [('', ['IQA_pytorch/weights/LPIPSvgg.pt','IQA_pytorch/weights/DISTS.pt'])],
    include_package_data=True,
    author='Keyan Ding',
    author_email='dingkeyan93@outlook.com',
    install_requires=["torch>=1.0"],
    url='https://github.com/dingkeyan93/IQA-pytorch',
    keywords = ['pytorch', 'similarity', 'IQA','metric','image-quality'], 
    platforms = "python",
    license='MIT',
)

# python setup.py sdist bdist_wheel
# twine check dist/*
# twine upload dist/*
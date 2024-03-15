from distutils.core import setup

# Python packaging done using instructions from
# https://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/index.html
setup(
    name='TBD',
    version='0.1dev',
    author='multiple (see github)',
    author_email='liuba.orlov.savko@rice.edu',
    packages=[
        'latent_active_learning',
    ],
    url='TBD',
    license='TBD',
    description='Shared repo for Latent Active Learning.',
    long_description=open('README.txt').read()
)
import os.path
import sys
try:
    from setuptools import setup
except ImportError:
    print("setuptools is required during the installation.\n")
    sys.exit(1)

AUTHOR = 'Wang Yao'
AUTHOR_EMAIL = 'wang.yao@honeywell.com'
CLASSIFIER = [
        'Intended Audience :: Data Science/Research',
        'Programming Language :: Python :: 3.6.5',
        'Topic :: Machine Learning :: Feature Selection',
        'Topic :: Software Development',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix', ]
DISTNAME = 'SupervisedSegmentation'
DESCRIPTION = 'A supervised segmentation module.'
KEYWORDS = 'segmentation'
LICENSE = 'LICENSE.txt'
MAINTAINER = 'Wang Yao'
MAINTAINER_EMAIL = 'wang.yao@honeywell.com'
PACKAGES = [
        'supervisedsegmentation',
        'supervisedsegmentation.test']
PACKAGE_DATA = {'': ['*']}
REQUIREMENTS = [
        "pydotplus",
        "pyodbc",
        "pandas >= 0.23.0",
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn"]
SCRIPTS = [
        'supervisedsegmentation/supervisedsegmentation.py',
        'supervisedsegmentation/CommonModules/dataloader.py',
        'supervisedsegmentation/CommonModules/utils.py',
        'supervisedsegmentation/CommonModules/loggerinitializer.py',
        'supervisedsegmentation/CommonModules/errors.py']
URL = 'https://'
VERSION = '0.1.0'


def readme():
    """Open 'README.md'

    :return: Content in 'README.md'
    """
    with open('README.md') as f:
        return f.read()


if __name__ == '__main__':
    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    setup(
        name=DISTNAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=readme(),
        classifiers=CLASSIFIER,
        keywords=KEYWORDS,
        author=AUTHOR,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        author_email=AUTHOR_EMAIL,
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        scripts=SCRIPTS,
        url=URL,
        license=LICENSE,
        install_requires=REQUIREMENTS,
        include_package_data=True,
        zip_safe=False
    )

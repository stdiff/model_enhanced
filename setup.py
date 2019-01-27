from setuptools import setup

setup(
    name="model_enhanced",
    version="0.2",
    description="Just helper classes for MLflow",
    url="https://github.com/stdiff/model_enhanced",
    author="Hironori Sakai",
    author_email="crescent.lab@gmail.com",
    license="MIT",
    packages=["model_enhanced"],
    python_requires=">=3.5",
    install_requires = ["pandas>=0.19.0",
                        "scikit-learn>=0.18.0"
                        ],
    zip_safe=False
)
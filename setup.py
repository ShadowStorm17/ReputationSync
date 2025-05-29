from setuptools import setup, find_packages

setup(
    name="reputation_sync",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "python-multipart>=0.0.6",
        "python-dotenv>=1.0.0",
        "jinja2>=3.1.2",
        "aiosqlite>=0.19.0",
        "aiofiles>=23.2.1",
    ],
) 
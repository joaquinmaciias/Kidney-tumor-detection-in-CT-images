import setuptools

setuptools.setup(
    name="nnunet",
    version="0.1.0",
    packages=["nnunetv2", "nnUNet_data"],
    # O, si tienes subpaquetes, puedes incluirlos con:
    # packages=setuptools.find_packages(include=["nnunetv2", "nnunetv2.*", "nnUNet_data", "nnUNet_data.*"]),
    # Agrega otros metadatos según necesites
)

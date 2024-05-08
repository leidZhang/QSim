import os


__rtModelDirPath = os.environ['RTMODELS_DIR']

# QCar RT Models
QCAR = os.path.normpath(
    os.path.join(__rtModelDirPath, 'qcar/QCar_Workspace'))

QCAR_STUDIO = os.path.normpath(
    os.path.join(__rtModelDirPath, 'qcar/QCar_Workspace_Studio'))


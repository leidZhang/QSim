import os


__imagesDirPath = os.path.normpath(
    os.path.join(os.environ['QAL_DIR'], 'libraries/resources/images/')
)


SDCS_CITYSCAPE = os.path.normpath(
    os.path.join(__imagesDirPath, 'sdcs_cityscape.png'))

SDCS_CITYSCAPE_SMALL = os.path.normpath(
    os.path.join(__imagesDirPath, 'sdcs_cityscape_small.png'))

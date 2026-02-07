# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('processphotos.py', '.'), ('logo.png', '.'), ('C:\\Users\\Abdullah Khawaja\\.insightface\\models\\buffalo_l', 'insightface_models/buffalo_l')]
binaries = []
hiddenimports = ['PIL', 'PIL.Image', 'PIL._tkinter_finder', 'cv2', 'numpy', 'numpy.core._methods', 'numpy.lib.format', 'onnxruntime', 'onnxruntime.capi', 'insightface', 'insightface.app', 'insightface.app.face_analysis', 'insightface.model_zoo', 'insightface.model_zoo.model_zoo', 'insightface.utils', 'insightface.utils.face_align', 'sklearn', 'sklearn.cluster', 'sklearn.neighbors', 'scipy', 'scipy.spatial', 'scipy.special', 'albumentations', 'prettytable', 'easydict']
tmp_ret = collect_all('insightface')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('onnxruntime')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('cv2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['photo_organizer.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SortMoments',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['C:\\Users\\Abdullah Khawaja\\Downloads\\PhotoOrganizerApp\\logo.ico'],
)

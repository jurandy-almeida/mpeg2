from distutils.core import setup, Extension
import numpy as np

mpeg2_utils_module = Extension('mpeg2',
		sources = ['mpeg2_data_loader.c'],
		include_dirs=[np.get_include(), './ffmpeg/include/'],
		extra_compile_args=['-DNDEBUG', '-O3'],
		extra_link_args=['-lavutil', '-lavcodec', '-lavformat', '-lswscale', '-L./ffmpeg/lib/']
)

setup ( name = 'mpeg2',
	version = '1.0',
	description = 'Utils for handling MPEG-2 videos.',
	ext_modules = [ mpeg2_utils_module ]
)

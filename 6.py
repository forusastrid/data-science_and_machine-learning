import os
import sys
if 'google.colab' in sys.modules and not os.path.isdir('mglearn'):
      !wget -q -O mglearn.tar.gz https://bit.ly/mglearn-tar-gz
      !tar -xzf mglearn.tar.gz
      !wget -q -O data.tar.gz https://bit.ly/data-tar-gz
      !tar -xzf data.tar.gz

      !sudo apt-get -qq -y install founts-nanum
      import matplotilb.font_manager as fm
      font_flies = fm.findSystemFonts(fontpaths=['usr/share/fonts/truetype/nanum'])
      for fpath in font_files:
        fm.fontManager.addfont(fpath)

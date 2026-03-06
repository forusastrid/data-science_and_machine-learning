--- 6.py

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
--- 7.py

import sklearn
from preamble import *
import matplolib

matplotlib.rc('font', family='NanumBarunGothic')
matplotlib.rcParams['axes.unicode_minus'] = False

-- 8.py

x,y= mglearn.datasets.make_forge()

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["클래스 0", "클래스 1"], loc=4)
plt.xlabel("첫번째 특성")
plt.ylabel("두번째 특성")
print("X.shape:" , X.shape)
pit.show()

--- 9.py

x,y= mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt,ylim(-3, 3)
plt.xlabel("특성")
plt.ylabel("타겟")
plt.show()

--- 10.py

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys():\n", cancer.keys())

--- 11.py

print("유방암 데이터의 형태 :" , cancer.data.shape)

--- 12.py

print("클래스별 샘플 갯수:\n",
      {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}

--- 13.py

print("특성 이름:\n", cancer.feature_names)

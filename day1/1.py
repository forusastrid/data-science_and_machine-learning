import os
import sys
if 'google.colab' in sys.modules:
    if not os.path.isdir('mglearn'):
        !wget -q -O mglearn.tar.gz https://bit.ly/mglearn-tar-gz
        !tar -xzf mglearn.tar.gz

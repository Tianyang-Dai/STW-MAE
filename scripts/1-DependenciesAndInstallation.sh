#!/bin/bash
# 依赖项和安装

# 删除环境
# conda deactivate
# conda remove --name STW-MAE --all
# conda env list

# 创建环境
# conda create --name STW-MAE python=3.9 -y
# conda activate STW-MAE
# source activate STW-MAE

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install absl-py==1.2.0
pip install backoff==2.1.2
pip install Bottleneck
pip install brotlipy==0.7.0
pip install bytecode==0.15.1
pip install cachetools==5.2.0
pip install certifi
pip install cffi
pip install charset-normalizer
pip install click==8.1.3
pip install contourpy==1.0.6
pip install cryptography
pip install cycler==0.11.0
pip install DateTime==4.7
pip install docker-pycreds==0.4.0
pip install filelock==3.13.1
pip install fonttools==4.38.0
pip install fsspec==2023.12.2
pip install gitdb==4.0.9
pip install GitPython==3.1.27
pip install google-auth==2.12.0
pip install google-auth-oauthlib==0.4.6
pip install gql==3.4.0
pip install graphql-core==3.2.3
pip install grpcio==1.49.1
pip install huggingface-hub==0.20.1
pip install idna
pip install joblib
pip install kiwisolver==1.4.4
pip install Markdown==3.4.1
pip install MarkupSafe==2.1.1
pip install matplotlib==3.6.2
pip install mkl-fft==1.3.1
pip install mkl-random
pip install mkl-service==2.4.0
pip install multidict==6.0.2
pip install numexpr
pip install numpy
pip install nvidia-ml-py3==7.352.0
pip install oauthlib==3.2.1
pip install opencv-python==4.6.0.66
pip install packaging
pip install pandas==1.4.4
pip install pathlib
pip install pathtools==0.1.2
pip install Pillow==9.2.0
pip install platformdirs
pip install pooch
pip install promise==2.3
pip install protobuf==3.19.5
pip install psutil==5.9.2
pip install pyasn1==0.4.8
pip install pyasn1-modules==0.2.8
pip install pycparser
pip install pyOpenSSL
pip install pyparsing
pip install PySocks
pip install python-dateutil
pip install pytz
pip install PyYAML==6.0
pip install regex==2023.12.25
pip install requests
pip install requests-oauthlib==1.3.1
pip install rsa==4.9
pip install safetensors==0.4.1
pip install scikit-learn
pip install scipy==1.10.1
pip install sentry-sdk==1.9.9
pip install setproctitle==1.3.2
pip install shortuuid==1.0.9
pip install six
pip install smmap==5.0.0
pip install subprocess32==3.5.4
pip install tensorboard==2.10.1
pip install tensorboard-data-server==0.6.1
pip install tensorboard-plugin-wit==1.8.1
pip install threadpoolctl
pip install timm==0.3.2
pip install tokenizers==0.15.0
pip install tqdm==4.66.1
pip install transformers==4.36.2
pip install typing_extensions
pip install urllib3
pip install visualizer==0.0.1
pip install wandb==0.13.3
pip install watchdog==2.1.9
pip install Werkzeug==2.2.2
pip install yarl==1.8.1
pip install zope.interface==5.4.0
pip install xlrd
pip install PyWavelets

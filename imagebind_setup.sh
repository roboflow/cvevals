git clone https://github.com/facebookresearch/ImageBind.git

conda create --name imagebind python=3.8 -y
conda activate imagebind

pip3 install -r requirements.txt
cp examples/imagebind_example.py ImageBind/imagebind_example.py
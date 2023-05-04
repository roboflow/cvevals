pip3 install -r requirements.txt

git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/

pip3 install -r requirements.txt
pip3 install -e .

mkdir weights
cd weights

wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

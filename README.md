# Waste Detection in Real Time using DNN

## shared rules
- Labels: 0 = Landfill, 1 = Recycle, 2 = Compost
- Image size: 64x64, RGB, pixels scaled 0 to 1
- Data: separate train / val / test arrays (shared via Google Drive, not GitHub)
- predict(image) -> 3 probabilities [Landfill, Recycle, Compost]

## folders
- data/      malek: dataset prep, preprocessing
- model/     humraj: NumPy CNN, Keras reference
- optimize/  benson: pruning, quantization, Huffman
- demo/      benson: OpenCV webcam demo
- results/   plots and metrics

## Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Biometria - Projekt 1 - uwierzytelnianie na podstawie twarzy

### Informacje
* [polecenie](https://www.syga.ai.pwr.edu.pl/courses/bio/P1.pdf)
* [wykłady](https://www.syga.ai.pwr.edu.pl/courses/bio/lec.html)
  * [introduction](https://www.syga.ai.pwr.edu.pl/courses/bio/lec.html)
  * [wprowadzenie do biometrii](https://www.syga.ai.pwr.edu.pl/courses/bio/l01.pdf)
    [transformata Fouriera, Convolution theorem, Eigenfaces](https://www.syga.ai.pwr.edu.pl/courses/bio/l03.pdf)
* [raport](./raport.pdf)
### Uruchomienie
```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
git submodule update --init --recursive
# bash models/arc_face/setup.sh
# download dataset
python3 preprocess_dataset.py
python3 dataset.py
```
### Rozważane modele
| model | source | submodule |
|:-|:-:|:-:|
|ArcFace | [source](https://github.com/ronghuaiyang/arcface-pytorch) | [submodule](./models/arc_face/) |
|DeepFace | [source](https://github.com/serengil/deepface) | [submodule](./models/deep_face/) |
|GhostFace | [source](https://github.com/Hazqeel09/ellzaf_ml#ghostfacenets) | [submodule](./models/ghost_face/) |
|InsightFace | [source](https://github.com/TreB1eN/InsightFace_Pytorch) | [submodule](./models/insight_face/) |

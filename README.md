# Biometria - Projekt 1 - uwierzytelnianie na podstawie twarzy

### Informacje
* [polecenie](https://www.syga.ai.pwr.edu.pl/courses/bio/P1.pdf)
* [wykłady](https://www.syga.ai.pwr.edu.pl/courses/bio/lec.html)
  * [introduction](https://www.syga.ai.pwr.edu.pl/courses/bio/lec.html)
  * [wprowadzenie do biometrii](https://www.syga.ai.pwr.edu.pl/courses/bio/l01.pdf)
    [przypomnienie zagadnienień z zakresu przetwarzania obrazu](https://www.syga.ai.pwr.edu.pl/courses/bio/l02.pdf)

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

## [ArcFace](https://github.com/ronghuaiyang/arcface-pytorch)
## [DeepFace](https://github.com/serengil/deepface)
## [GhostFace](https://github.com/Hazqeel09/ellzaf_ml#ghostfacenets)
## [InsightFace](https://github.com/TreB1eN/InsightFace_Pytorch)

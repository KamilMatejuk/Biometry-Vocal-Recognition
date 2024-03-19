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
bash models/arc_face/setup.sh
# download dataset
python3 preprocess_dataset.py
python3 dataset.py
```

# Inne
system autoryzacji użytkownika na podstawie twarzy
- dodanie nowych użytkowników
- autoryzacja
modele: InsightFace, GhostFace, DeepFace, ArcFace - trening lub finetuning, zależnie od modelu chyba
zbiór FaceScrub lub CelebA (osobno trenigowy, walidacyjny, testowy i wdrożony do bazy użytkowników ???)
dostępny preprocessing



wzrucamy zdjecia do bazy, a potem model nam mowi czy jest podobny do kogos w bazie czy nie ???


## [InsightFace](https://github.com/TreB1eN/InsightFace_Pytorch)
* zdjęcia ludzi trzymane w bazie - po prostu pliki
* model tworzy reprezentacje ludzi
  * za każdym razem czy tylko raz ?
* dopasowywana jest twarz do bazy i zwracana jej pozycja
  * chyba po prostu czy jest ktoś w bazie, a nie kto ?
  * co jak preternowane, czy może znaleźć kogoś kto był w datasecie treningowym ?
* nie wymaga preprocessingu zdjęć

## [GhostFace](https://github.com/Hazqeel09/ellzaf_ml#ghostfacenets)
* jest metoda która zwraca klasyfikację i embeddingi
  * wytrenować na klasyfikacji, przetestować thresholdy, autentykować na reprezentacji
* nwm czy są pretrenowane wagi

## [DeepFace](https://github.com/serengil/deepface)
* da się porównać 2 zdjęcia, czy to ta sama osoba
  * trzeba by przerobić, żeby korzystało z zapisanych reprezentacji, a nie zdjęć
* domyślnie wrapper na VGG-Face, ale może być też ArcFace lub GhostFaceNet
  * wykorzytsuje modele w tensorflow
* weryfikacja: https://github.com/serengil/deepface/blob/master/deepface/modules/verification.py#L16
* nwm czy da się wygodnie dotrenować

## [ArcFace](https://github.com/ronghuaiyang/arcface-pytorch)
* prosta implementacja na Resnecie
  * nwm jeszcze które to reprezentacja
* są dostępne wagi
* repo sprzed 6 lat


# Pipe line
* download dataset
* split into train, test, validate, db
* add our images to db
* for each model:
  * define dataloader with transfromations
  * train and validate metrics
  * get representation from image
* create representations of db in vector database
* authenticate user


spróbować uruchomić każdy model, załadować wagi, pobrać reprezentację i dotrenować
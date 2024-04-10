# Biometria - Projekt 2 - uwierzytelnianie na podstawie głosu

## Setup
1. Inicjalizacja środowiska
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Dodanie wykorzystywanego modelu
```
git submodule add <model-url> models/<model-name>/src
```
3. Dodanie loggera w pliku `loggers.py` (podobnie jak `example_logger`)
4. Stworzenie nakładki na model w pliku `models/<model-name>/__init__.py` (zgodnie z szablonem z `models/example/__init__.py`)
   1. Parametry do finetuningu można dodać do pliku `config.yml`
5. [Opcjonalnie] Dodanie skryptu `models/<model-name>/setup.sh` który np zmieni importy w `src` na relatywne (zobacz `models/example/setup.sh`)
6. Pobranie datasetu do folderu `data/inputs/images` oraz `data/inputs/identities.txt` zawierające nazwę pliku i klasę/użytkownika
7. [Opcjinalnie] Aby dodać szum albo inne zniekształcenia do datasetu, można zainspirować się `dataset_modification.py`
8. Przerobić `dataset.py` aby działał z dźwiękiem (ten plik powinien wygenerować pliki zawierające odpowiednie partycje)
9. Do testów hiperparametrów można posłużyć się plikiem `run.sh`
10. Otworzyć UI w przegldarce jako plik `ui_interface.html` i uruchomić serwer
```
python -m uvicorn server:app --host 127.0.0.1 --port 8080
```


## Struktura danych
### Folder `models/`
```bash
models/                   # folder z modelami
|---example/              # folder z konkretnym modelem
|   |---src/              # folder z kodem źródłowym modelu dodanym jako submodule
|   |---__init__.py       # nakładka na kod źródłowy modelu
|---__init__.py           # interface nakładki
```
### Folder `data/`
```bash
data/                     # folder z danymi (in & out)
|---checkpoints/          # folder z zapisanymi modelami
|   |---example/          # dla modelu example
|       |---name.chpt     # dla eksperymentu o nazwie <name>
|---embeddings/           # folder z zapisanymi reprezentacjami wektorowymi danych
|   |---example/          # dla modelu example
|       |---name.db       # dla basy danych stworzonej na podstawie eksperymentu o nazwie <name>
|---inputs/               # folder z danymi wejściowymi
|   |---images/           # zwykłe dane wejściowe
|   |---images_noise/     # dane wejściowe po nałożeniu np szumu
|   |---identities.txt    # plik z rozpisanymi przypisaniami klasy do pliku w images/
|   |---partition_test.db # plik z rozpisanymi przypisaniami klasy do pliku w images/ dla konkretnego zbioru (testowy, treningowy, etc)
|---metrics/              # folder z wynikami eksperymentów (loss, accuracy, etc)
    |---example/          # dla modelu example
        |---name.csv      # dla eksperymentu o nazwie <name>
```

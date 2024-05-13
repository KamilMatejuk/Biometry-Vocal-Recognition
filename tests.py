from database import Distance,  get_db, auth
from loggers import main_logger as logger
from sounds import alter_amplitude, subsample, add_noise, add_noise_from_file
import random
import wespeaker 
import os
import numpy as np

#  1. Przetestowanie skuteczności systemu (wyrażonej w omówionych metrykach) na nie mniej niż 500 próbkach wdrożonych użytkowników (zbalansuj prawidłowe próby uwierzytelnienia i próby podszycia się pod innych użytkowników).
def test1(model_name, data_folder: str, distance: Distance = Distance.EUCLIDEAN):
    model = wespeaker.load_model(model_name)

    result_true = 0
    result_false = 0

    for folder in os.listdir(data_folder):
        folder_path = f'{data_folder}/{folder}'
        if not os.path.isdir(folder_path):
            continue

        user_id = folder.replace('id', '')
        
        for inner_folder in os.listdir(folder_path):
            inner_folder_path = f'{folder_path}/{inner_folder}'
            if not os.path.isdir(inner_folder_path):
                continue

            for file in os.listdir(inner_folder_path):
                embedding = model.extract_embedding(f'{inner_folder_path}/{file}')

                if auth(model_name, embedding, f'user_{user_id}', distance):
                    result_true += 1
                else:
                    result_false += 1

    total = result_true + result_false
    logger.info(f'Test1 TRUE: {result_true} FALSE: {result_false} Accuracy: {result_true/total}')

# 2. Dla podzbioru przynajmniej 500 próbek wykonaj losowo (z prawdopodobieństwem jednostajnym) przemnożenia amplitudy próbki przez wartość ze zbioru
# {25, 1, 0.04} a następnie porównaj skuteczność modelu z wynikami uzyskanymi
# w zadaniu 1.
def test2(model_name, data_folder: str, distance: Distance = Distance.EUCLIDEAN):
    model = wespeaker.load_model(model_name)

    multipliers = [25, 1, 0.04]
    result_true = 0
    result_false = 0

    for folder in os.listdir(data_folder):
        folder_path = f'{data_folder}/{folder}'
        if not os.path.isdir(folder_path):
            continue

        user_id = folder.replace('id', '')
        
        for inner_folder in os.listdir(folder_path):
            inner_folder_path = f'{folder_path}/{inner_folder}'
            if not os.path.isdir(inner_folder_path):
                continue

            for file in os.listdir(inner_folder_path):
                file_path = f'{inner_folder_path}/{file}'

                multiplier = random.choice(multipliers)
                altered_file = alter_amplitude(file_path, multiplier, 'temp2.wav')

                embedding = model.extract_embedding(altered_file)

                if auth(model_name, embedding, f'user_{user_id}', distance):
                    result_true += 1
                else:
                    result_false += 1

    total = result_true + result_false
    logger.info(f'Test2 TRUE: {result_true} FALSE: {result_false} Accuracy: {result_true/total}')

# 3. Dla wybranych 200 próbek sygnału, zmniejsz częstotliwość próbkowania poprzez
# pozostawienie co 2, co 5 oraz co 10 wartości (pamiętaj o metadanych). Sprawdź
# jak subsampling wpływa na wymaganą długość próbki sygnału oraz na skuteczność uwierzytelniania.
def test3(model_name, data_folder: str, subsampling: int, count: int = 200, distance: Distance = Distance.EUCLIDEAN):
    model = wespeaker.load_model(model_name)

    result_true = 0
    result_false = 0

    counter = 0

    for folder in os.listdir(data_folder):
        folder_path = f'{data_folder}/{folder}'
        if not os.path.isdir(folder_path):
            continue

        user_id = folder.replace('id', '')
        
        for inner_folder in os.listdir(folder_path):
            inner_folder_path = f'{folder_path}/{inner_folder}'
            if not os.path.isdir(inner_folder_path):
                continue

            for file in os.listdir(inner_folder_path):
                file_path = f'{inner_folder_path}/{file}'

                altered_file = subsample(file_path, subsampling, 'temp3.wav')

                embedding = model.extract_embedding(altered_file)

                if auth(model_name, embedding, f'user_{user_id}', distance):
                    result_true += 1
                else:
                    result_false += 1

                counter += 1

                if counter == count:
                    break

            if counter == count:
                break
        
        if counter == count:
            break

    total = result_true + result_false
    logger.info(f'Test3 TRUE: {result_true} FALSE: {result_false} Accuracy: {result_true/total}')

# 4. Dodaj do 100 próbek zakłócenia o rozkładzie {N (0, 1), N (0, 10), N (5, 1)}, a
# następnie zbadaj skuteczność systemu.
def test4(model_name, data_folder: str, n_first: int, n_second: int, count: int = 100, distance: Distance = Distance.EUCLIDEAN):
    model = wespeaker.load_model(model_name)

    result_true = 0
    result_false = 0

    counter = 0

    for folder in os.listdir(data_folder):
        folder_path = f'{data_folder}/{folder}'
        if not os.path.isdir(folder_path):
            continue

        user_id = folder.replace('id', '')
        
        for inner_folder in os.listdir(folder_path):
            inner_folder_path = f'{folder_path}/{inner_folder}'
            if not os.path.isdir(inner_folder_path):
                continue

            for file in os.listdir(inner_folder_path):
                file_path = f'{inner_folder_path}/{file}'

                altered_file = add_noise(file_path, n_first, n_second, 'temp4.wav')

                embedding = model.extract_embedding(altered_file)

                if auth(model_name, embedding, f'user_{user_id}', distance):
                    result_true += 1
                else:
                    result_false += 1

                counter += 1

                if counter == count:
                    break

            if counter == count:
                break
        
        if counter == count:
            break

    total = result_true + result_false
    logger.info(f'Test4 TRUE: {result_true} FALSE: {result_false} Accuracy: {result_true/total}')

# 5. Utwórz lub pobierz plik z nieregularnymi zakłóceniami (np. odgłosy psów), zmniejsz jego amplitudę, tak by maksymalna amplituda zakłóceń była połową maksymalnej amplitudy oryginalnego sygnału, a następnie dodaj zakłócenia do próbek 100 próbek. Przetestuj system uwierzytelniania na tak zakłóconych próbkach.
def test5(model_name, data_folder: str, noise_sound_file: str, count: int = 100, distance: Distance = Distance.EUCLIDEAN):
    model = wespeaker.load_model(model_name)

    result_true = 0
    result_false = 0

    counter = 0

    for folder in os.listdir(data_folder):
        folder_path = f'{data_folder}/{folder}'
        if not os.path.isdir(folder_path):
            continue

        user_id = folder.replace('id', '')
        
        for inner_folder in os.listdir(folder_path):
            inner_folder_path = f'{folder_path}/{inner_folder}'
            if not os.path.isdir(inner_folder_path):
                continue

            for file in os.listdir(inner_folder_path):
                file_path = f'{inner_folder_path}/{file}'

                altered_file = add_noise_from_file(file_path, noise_sound_file, 'temp5.wav')

                embedding = model.extract_embedding(altered_file)

                if auth(model_name, embedding, f'user_{user_id}', distance):
                    result_true += 1
                else:
                    result_false += 1

                counter += 1

                if counter == count:
                    break

            if counter == count:
                break
        
        if counter == count:
            break

    total = result_true + result_false
    logger.info(f'Test5 TRUE: {result_true} FALSE: {result_false} Accuracy: {result_true/total}')
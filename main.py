import torch
import argparse
from database import Distance, add, get_similar, save_db_raw, save_db, auth
from tests import test1, test2, test3, test4, test5
from loggers import main_logger as logger
import wespeaker 
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def action_init_db(model_name, data_folder: str):
    model = wespeaker.load_model(model_name)

    vectors = {}
    vectors_raw = {}

    counter = 0
    users_total = len(os.listdir(data_folder))
    for folder in os.listdir(data_folder):
        folder_path = f'{data_folder}/{folder}'
        if not os.path.isdir(folder_path):
            continue

        user_id = folder.replace('id', '')
        user_vectors = []

        logger.info(f'Adding user {counter+1}/{users_total}')
        
        for inner_folder in os.listdir(folder_path):
            inner_folder_path = f'{folder_path}/{inner_folder}'
            if not os.path.isdir(inner_folder_path):
                continue

            for file in os.listdir(inner_folder_path):
                embedding = model.extract_embedding(f'{inner_folder_path}/{file}')

                user_vectors.append(embedding)

        vectors[f'user_{user_id}'] = torch.mean(torch.stack(user_vectors), dim=0)
        vectors_raw[f'user_{user_id}'] = user_vectors
        counter += 1

    logger.info(f'Added {counter} users')

    save_db_raw(model_name, vectors_raw)
    save_db(model_name, vectors)
    

def action_add(model_name: str, sound_file: str, user_id: str = "unknown"):
    model = wespeaker.load_model(model_name)

    embedding = model.extract_embedding(f'{sound_file}')
    add(f'{model_name}', f'user_{user_id}', embedding)

def action_identify(model_name: str, sound_file: str, distance: Distance = Distance.EUCLIDEAN):
    model = wespeaker.load_model(model_name)

    embedding = model.extract_embedding(f'{sound_file}')

    return get_similar(model_name, embedding, distance)

def action_auth(model_name: str, sound_file: str, user_id: str, distance: Distance = Distance.EUCLIDEAN):
    model = wespeaker.load_model(model_name)

    embedding = model.extract_embedding(f'{sound_file}')

    return auth(model_name, embedding, f'user_{user_id}', distance)


if __name__ == '__main__':
    initial_db_dir = 'sound_data'
    model_name = "english"

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', choices=['init_db', 'add', 'auth', 'identify', 'test1', 'test2', 'test3', 'test4', 'test5'], required=True, help='What to do?')
    parser.add_argument('-u', '--user_id', required=False, help='User label to be used with the "add action" -> "-a add"')
    parser.add_argument('-s', '--sound_file', required=False, help='Path to the sound file to be used with the "add action" -> "-a add"')
    parser.add_argument('-v', '--variant', choices=['a', 'b', 'c'], required=False, help='Varint used for tests')
    args = parser.parse_args()
    
    if args.action == 'init_db': action_init_db(model_name, initial_db_dir)
    elif args.action == 'add': action_add(model_name, args.sound_file, args.user_id)
    elif args.action == 'auth': action_auth(model_name, args.sound_file, args.user_id)
    elif args.action == 'identify': action_identify(model_name, args.sound_file)

    elif args.action == 'test1': test1(model_name, 'sound_data_test', Distance.MANHATTAN if args.variant == 'b' else (Distance.COSINE if args.variant == 'c' else Distance.EUCLIDEAN))
    elif args.action == 'test2': test2(model_name, 'sound_data_test', Distance.MANHATTAN if args.variant == 'b' else (Distance.COSINE if args.variant == 'c' else Distance.EUCLIDEAN))
    elif args.action == 'test3': test3(model_name, 'sound_data_test', 5 if args.variant == 'b' else (10 if args.variant == 'c' else 2))
    elif args.action == 'test4': test4(model_name, 'sound_data_test', 5 if args.variant == 'c' else 0, 10 if args.variant == 'b' else 1)
    elif args.action == 'test5': test5(model_name, 'sound_data_test', 'noise.wav', Distance.MANHATTAN if args.variant == 'b' else (Distance.COSINE if args.variant == 'c' else Distance.EUCLIDEAN))
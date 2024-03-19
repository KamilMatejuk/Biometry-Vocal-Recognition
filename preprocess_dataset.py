import os
import tqdm
from PIL import Image


if __name__ == '__main__':
    root_dir = 'data/inputs'
    size = 512

    os.makedirs(os.path.join(root_dir, 'images_resized'), exist_ok=True)
    for img_name in tqdm.tqdm(os.listdir(os.path.join(root_dir, 'images'))):
        img = Image.open(os.path.join(root_dir, 'images', img_name))
        width, height = img.size
        aspect_ratio = width / height
        new_width = size if width > height else int(size * aspect_ratio)
        new_height = int(size / aspect_ratio) if width > height else size
        image_pil_resized = img.resize((new_width, new_height))
        bg = Image.new("RGB", (size, size), color="black")
        offset = ((size - new_width) // 2, (size - new_height) // 2)
        bg.paste(image_pil_resized, offset)
        bg.save(os.path.join(root_dir, 'images_resized', img_name))
        # break

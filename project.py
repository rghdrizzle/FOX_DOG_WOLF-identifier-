from duckduckgo_search import ddg_images
from fastcore.all import *
from fastdownload import download_url
from fastai.vision.all import *


def search(term,max_images=50):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image')

searches = 'fox','wolf','dogs'
path =Path('fox_or_wolf_or_dog')
from time import sleep

for i in searches:
    dest=(path/i)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest,urls=search(f'{i} photo'))
    sleep(10)
    download_images(dest,urls=search(f'{i} color photo'))
    sleep(10)
    download_images(dest,urls=search(f'{i} nose photo'))
    sleep(10)
    download_images(dest,urls=search(f'{i} in snow photo'))
    sleep(10)
    download_images(dest,urls=search(f'{i} tail photo'))
    resize_images(path/i, max_size=400, dest=path/i)
    
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)


dls= DataBlock(
    blocks=(ImageBlock,CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y =parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=500)
dls.show_batch(max_n=200)

learn = vision_learner(dls,resnet18,metrics=error_rate)
learn.fine_tune(5)

dest = 'wolf1.jpg'
download_url('https://upload.wikimedia.org/wikipedia/commons/6/68/Eurasian_wolf_2.jpg', dest, show_progress=False)
im = Image.open(dest)
    
is_wolf,_,probs=learn.predict(PILImage.create('wolf1.jpg'))
print(f"This is a: {is_wolf}.")
print(f"Probability it's a wolf: {probs[2]:.4f}")

is_fox,_,probs=learn.predict(PILImage.create('fox.jpg'))
print(f"This is a: {is_fox}.")
print(f"Probability it's a fox: {probs[1]:.4f}")

is_dogs,_,probs=learn.predict(PILImage.create('fox.jpg'))
print(f"This is a: {is_dogs}.")
print(f"Probability it's a dog: {probs[0]:.4f}")
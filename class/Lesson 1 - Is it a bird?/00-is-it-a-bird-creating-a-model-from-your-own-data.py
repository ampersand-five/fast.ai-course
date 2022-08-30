from duckduckgo_search import ddg_images
# fastai.vision.all must be imported before fastcore.all on Intel Macs. Kills kernels
# otherwise, don't know why (as of Aug. 2022, might have a fix later)
from fastai.vision.all import *
from fastcore.all import *

def search_images(term, max_images=30):
    print(f"Searching for {max_images} '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image')

#NB: `search_images` depends on duckduckgo.com, which doesn't always return correct responses.
#    If you get a JSON error, just try running it again (it may take a couple of tries).
urls = search_images('bird photos', max_images=1)

from fastdownload import download_url
dest = 'data/single/bird.jpg'

download_url(urls[0], dest, show_progress=True)

im = Image.open(dest)
im.to_thumb(256,256)

download_url(search_images('forest photos', max_images=1)[0], 'data/single/forest.jpg', show_progress=True)
Image.open('data/single/forest.jpg').to_thumb(256,256)

searches = 'forest','bird'
path = Path('data/bird_or_not')
from time import sleep

num_images = 30

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    
    download_images(dest, urls=search_images(f'{o} photo', max_images=num_images))
    print(f'Downloaded {num_images} {o} photo')
    print('Sleeping 10 seconds to not overload server')
    sleep(10)  # Pause between searches to avoid over-loading server
    
    download_images(dest, urls=search_images(f'{o} sun photo', max_images=num_images))
    print(f'Downloaded {num_images} {o} sun photo')
    print('Sleeping 10 seconds to not overload server')
    sleep(10)
    
    download_images(dest, urls=search_images(f'{o} shade photo', max_images=num_images))
    print(f'Downloaded {num_images} {o} shade photo')
    print('Sleeping 10 seconds to not overload server')
    sleep(10)
    
    print(f'Resizing photos all {o} downloaded photos')
    resize_images(path/o, max_size=400, dest=path/o)

failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)

data_loaders = DataBlock(
    #       input type, output type -> fastai can figure out the model you need with these specified
    blocks=(ImageBlock, CategoryBlock),
    # function used to find items to train on
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    # function used to get correct labels for each data point
    get_y=parent_label,
    # item transforms
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

# Fast AI learners combine models with data
learn = vision_learner(data_loaders, resnet18, metrics=error_rate)
learn.fine_tune(3)

prediction,_,confidence = learn.predict(PILImage.create('data/single/bird.jpg'))
print(f"This is a: {prediction}.")
print(f"Confidence it's a bird: {confidence[0]:.4f}")

# Export model
model_path = Path('models')
model_path.mkdir(exist_ok=True, parents=True)
learn.export(model_path/'resnet18.pkl')
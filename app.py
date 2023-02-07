import gradio as gr

from duckduckgo_search import ddg_images
from fastcore.all import *
from fastai.vision.all import *
import gradio as gr


def search(term,max_images=50):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image')

learn=load_learner('model.pkl')

categories = ('dogs','fox','wolf')

def classify_image(img):
    pred,idx,probs = learn.predict(PILImage.create(img))
    return dict(zip(categories, map(float,probs)))

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples=['dog.jpg','fox.jpg','catdog.jpg','wolf.jpg','dog2.jpg','wolf1.jpg']

intf= gr.Interface(fn=classify_image ,inputs=image,outputs=label,examples=examples)
intf.launch(inline=False)
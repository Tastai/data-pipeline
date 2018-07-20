from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import requests
import numpy as np
import json
import spacy
import flask
from flask import jsonify
import io

app = flask.Flask(__name__)
model = None
nlp = spacy.load('en_core_web_lg')
ingreds_path = './ingredients_raw.json'

PROB_THRESH = .40
SIM_THRESH = .65

kitchen_docs = None # Kitchenware
ingredient_docs = None # Ingredients



def prep_docs(vocab):
    return [nlp(w) for w in vocab]

def prepKitchenDocs():
  global kitchen_docs 
  kitchen_vocab = ["plates","utensils","forks","knives","spoons","measuring spoons","tongs","spatula","whisk","pots","pans","cups","measuring cup","cutting board","baking dish","beater","blender","bottle opener","bowl","cake pan","cookie cutter","crockpot","cup","egg beater","timer","food processor","frying pan","garlic press","grater","griddle","grill","grinder","ice box","ice bucket","ice cream scoop","ice cube tray","jar","jug","juicer","kettle","knife","knife sharpener","ladle","lid","masher","meat grinder","meat thermometer","microwave","mixer","mixing bowl","mold","muffin pan","mug","mortar and pestle","nut cracker","nut pick","opener","oven mitt","pan","pepper grinder","pepper shaker","pitcher","pizza cutter","pizza pan","plate","platter","pot","potato masher","pressure cooker","quiche pan","ramekin","rice cooker","roaster","roasting pan","rolling pin","salad bowl","salt shaker","sauce voat","cause pan","saucer","sifter","skewers","slow cooker","skillet","slicer","spatula","spice jar","spoon","steak knife","steamer","stew pot","stove","tabespoon","tea cup","tea infuser","teapot","toaster","tray","waffle iron","whip","wok","zester"]
  kitchen_docs = prep_docs(kitchen_vocab)

def prepIngredientDocs():
  global ingredient_docs
  all_ingreds = json.load(open(ingreds_path))['ingred_data']
  ingredients_sample = all_ingreds[:500]
  ingreds_pre = [x['ingredients'] for x in ingredients_sample]

  ingreds = set()
  for ingred_list in ingreds_pre:
      for ingred in ingred_list:
          ingreds.add(ingred)
          
  ingredient_docs = prep_docs(list(ingreds))


def detSetSim(doc, wordSet):
    top_sim = 0;
    for item in wordSet:
        sim = item.similarity(doc)
        if sim > top_sim:
            top_sim = sim
        
    return top_sim


def determineCategory(doc):
    kitchen_sim = detSetSim(doc, kitchen_docs)
    ingred_dim = detSetSim(doc, ingredient_docs)
    
    if kitchen_sim < SIM_THRESH and ingred_dim < SIM_THRESH:
        return None
    
    if kitchen_sim < ingred_dim:
        return "ingredient"
    else:
        return "kitchenware"

def determineRelevance(labels):
    top_prob = 0
    item_label = None
    item_cat = None
    
    for item in labels:
        if item['probability'] < PROB_THRESH:
            continue
        
        top_prob = item['probability']
        
        label = item['label']
        label = label.replace('_', ' ')
        label_doc = nlp(label)
        
        cat = determineCategory(label_doc)
        if cat == None:
            continue
        else:
            item_label = label
            item_cat = cat
    
    if top_prob < PROB_THRESH or item_cat == None:
        if top_prob < PROB_THRESH:
            print("under prob thresh", top_prob)
            
        if item_cat == None:
            print("under similarity thesh")
        
        return {
            "is_relevant": False,
            "category": None,
            "possible_label": None 
        }
    
    return {
        "is_relevant": True,
        "category": item_cat,
        "possible_label": item_label 
    }
    
    

def load_model(): 
  global model
  model = ResNet50(weights='imagenet')  

def prepare_image(image, target):
  if image.mode != "RGB":
    image = image.convert("RGB")

  image = image.resize(target)
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  image = imagenet_utils.preprocess_input(image)

  return image


@app.route("/predict", methods=["POST"])
def predict():

  data = {
    "success": False
  }

  image = None
  if flask.request.method == "POST":
    if flask.request.files.get("image"):
      # read into PIL format
      image = flask.request.files["image"].read()
      image = Image.open(io.BytesIO(image))

    if flask.request.args.get('image_url'):
      img_url = flask.request.args.get('image_url')
      response = requests.get(img_url)
      image = Image.open(io.BytesIO(response.content))

    if image:
      # preprocess
      image = prepare_image(image, target=(224, 224))

      # classify
      preds = model.predict(image)
      results = imagenet_utils.decode_predictions(preds)
      data["predictions"] = []
      predictions = []

      for (imagenetId, label, prob) in results[0]:
        r = {
          "label": label,
          "probability": float(prob)
        }
        data["predictions"].append(r)
        predictions.append(r)

      relevance_result = determineRelevance(predictions)
      data['results'] = relevance_result

      data["success"] = True

  return flask.jsonify(data)


if __name__ == "__main__":
  print("Loading model and starting server...")
  load_model()
  prepKitchenDocs()
  prepIngredientDocs()
  app.run(host='0.0.0.0')


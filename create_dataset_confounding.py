import random
import json

random.seed(42)

categories = {
    "fruit": ["apple", "banana", "cherry", "grape", "orange", "kiwi", "mango", "pear", "peach"],
    "animal": ["dog", "cat", "lion", "tiger", "elephant", "bear", "wolf", "giraffe", "zebra"],
    "vehicle": ["car", "bus", "bike", "train", "plane", "truck", "scooter", "boat", "submarine"],
    "color": ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "pink"],
    "country": ["usa", "canada", "brazil", "china", "india", "germany", "france", "japan", "uganda"],
    "instrument": ["guitar", "piano", "violin", "drums", "flute", "trumpet", "saxophone"],
    "shape": ["circle", "square", "triangle", "rectangle", "oval", "hexagon", "pentagon"],
    "clothing": ["shirt", "pants", "dress", "skirt", "hat", "jacket", "shoes", "socks"]
}

distractors = ["bowl", "tree", "book", "lamp", "chair", "shoe", "pen", "window", "phone", "cup", "road"]

source_data = []

with open('diverse_count.json', 'r') as f:
    target_data = json.load(f)

for item in target_data:
    category = item['type']
    word_list = item['list']
    category_words = categories[category]
    matching_in_list = [w for w in word_list if w in category_words]
    available_distractors = [d for d in distractors if d not in word_list]
    word_to_replace = random.choice(matching_in_list)
    word_to_add = random.choice(available_distractors)
    idx = word_list.index(word_to_replace)
    item['list'][idx] = word_to_add
    item['answer'] -= 1
    source_data.append(item)


with open("confounding_data.json", "w") as f:
    json.dump(source_data, f)

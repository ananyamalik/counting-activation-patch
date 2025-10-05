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

dataset = []

for _ in range(10000):
    category = random.choice(list(categories.keys()))
    num_matching = random.randint(1, min(5, len(categories[category])))
    matching_words = random.sample(categories[category], num_matching)
    num_distractors = random.randint(2,10)
    non_matching_pool = distractors + [w for cat in categories for w in categories[cat] if cat != category]
    non_matching_words = random.sample(non_matching_pool, num_distractors)

    word_list = matching_words + non_matching_words
    random.shuffle(word_list)
    word_list = [w if random.random() < 0.2 else w for w in word_list]

    answer = sum(1 for w in word_list if w.lower() in [x.lower() for x in categories[category]])

    dataset.append({
        "type": category,
        "list": word_list,
        "answer": answer
    })

with open("diverse_count.json", "w") as f:
    json.dump(dataset, f)

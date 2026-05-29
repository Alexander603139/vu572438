import os, yaml, sys, re
from collections import defaultdict

# Определяем путь к YAML-файлу — он лежит рядом с этим скриптом
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MARKERS_PATH = os.path.join(BASE_DIR, 'political_markers.yml')

# Загрузка словаря
with open(MARKERS_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

categories = config['categories']
markers = {}
for cat, data in categories.items():
    markers[cat] = data['markers']

def classify_text(text):
    text_lower = text.lower()
    scores = defaultdict(int)
    total = 0

    for cat, words in markers.items():
        for word in words:
            count = len(re.findall(r'\b' + re.escape(word.lower()) + r'\b', text_lower))
            if count > 0:
                scores[cat] += count
                total += count

    if total == 0:
        return {cat: 0 for cat in markers}

    return {cat: round(scores[cat] / total * 100, 1) for cat in markers}

if __name__ == '__main__':
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    result = classify_text(text)
    print("\nРезультат классификации:")
    for cat, percent in result.items():
        name = categories[cat]['name']
        print(f"  {name}: {percent}%")
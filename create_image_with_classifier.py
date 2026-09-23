import pickle
import json
import struct
import zlib
from PIL import Image

import warnings # Stop simhash warning making my console messy.
warnings.filterwarnings('ignore', category=SyntaxWarning, message='invalid escape sequence')

model_orange = 'animals.pkcls'
model_json = 'model.json'
image_file = 'image/donegal_animals.png'
model_image = 'donegal_animals_model.png'

# Boxes around animal in images.
regions = {
  'sheep':   {'x': 0.095, 'y': 0.22, 'w': 0.32,  'h': 0.67},
  'rook':    {'x': 0.460, 'y': 0.40, 'w': 0.215, 'h': 0.58},
  'jackdaw': {'x': 0.635, 'y': 0.10, 'w': 0.068, 'h': 0.40},
  'cow':     {'x': 0.772, 'y': 0.00, 'w': 0.172, 'h': 1.00}
}

MAX_WIDTH = 2560 # WordPress swaps wider uploads for a resized copy, destroying my classifier. :/
MAGIC = b'FCM1' # Fun classifier model 1, magic number. :D
LENGTH_FORMAT = '>I' # Unsigned integer - big-endian


def get_bag_of_words_terms(attr):
    compute_value = attr.compute_value
    while not hasattr(compute_value, 'compute_shared'):
        compute_value = compute_value.variable.compute_value
    return compute_value


def create_model_from_orange_file(model):
    shared = get_bag_of_words_terms(model.domain.attributes[0]).compute_shared
    _, tokenizer, stopwords = shared.preprocessor.preprocessors

    vocab = [get_bag_of_words_terms(attr).name for attr in model.domain.attributes]
    coef = model.skl_model.coef_.round(8).tolist()

    classes = list(model.domain.class_var.values)
    global_weights = [1.0] * len(vocab)
    pattern = tokenizer._RegexpTokenizer__pattern
    remove_words = sorted(stopwords._StopwordsFilter__stopwords)
    intercept = [round(float(b), 8) for b in model.skl_model.intercept_]

    return {
        'format': 'funclassify.bow-linear/1',
        'classes': classes,
        'preprocess': [
            {'type': 'lowercase'},
            {'type': 'tokenize', 'pattern': pattern},
            {'type': 'remove_words', 'words': remove_words},
        ],
        'bow': {'term_frequency': 'count', 'global_weight': 'none', 'normalization': 'none'},
        'vocabulary': vocab,
        'global_weights': global_weights,
        'model': {
            'type': 'logistic_regression',
            'link': 'multinomial',
            'coef': coef,
            'intercept': intercept,
        },
    }


with open(model_orange, 'rb') as file:
    orange_model = pickle.load(file)

spec = create_model_from_orange_file(orange_model)

with open(model_json, 'w', encoding='utf-8') as file:
    json.dump(spec, file, ensure_ascii=False, indent=4)

print(f'Wrote "{model_json}", Classes: {spec["classes"]}, {len(spec["vocabulary"])} terms')

# Now start building the image.
spec['regions'] = regions

data = json.dumps(spec, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
blob = zlib.compress(data, 9)
payload = MAGIC + struct.pack(LENGTH_FORMAT, len(blob)) + blob

image = Image.open(image_file).convert('RGB')
if image.width > MAX_WIDTH:
    image = image.resize((MAX_WIDTH, round(image.height * MAX_WIDTH / image.width)), Image.LANCZOS)

pixels = bytearray(image.tobytes())
bits = [(byte >> shift) & 1 for byte in payload for shift in range(7, -1, -1)]
for i, bit in enumerate(bits):
    pixels[i] = (pixels[i] & 0xFE) | bit

Image.frombytes('RGB', image.size, bytes(pixels)).save(model_image, 'PNG', optimize=True)
print(f'Hid {len(blob):,} bytes in {model_image}')

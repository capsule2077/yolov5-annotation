import json

import cv2
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from torchvision.models.feature_extraction import create_feature_extractor
from matplotlib import pyplot as plt

label_class = class_indices = {
    0: 'house_finch',
    1: 'robin',
    2: 'triceratops',
    3: 'green_mamba',
    4: 'harvestman',
    5: 'toucan',
    6: 'goose',
    7: 'jellyfish',
    8: 'nematode',
    9: 'king_crab',
    10: 'dugong',
    11: 'Walker_hound',
    12: 'Ibizan_hound',
    13: 'Saluki',
    14: 'golden_retriever',
    15: 'Gordon_setter',
    16: 'komondor',
    17: 'boxer',
    18: 'Tibetan_mastiff',
    19: 'French_bulldog',
    20: 'malamute',
    21: 'dalmatian',
    22: 'Newfoundland',
    23: 'miniature_poodle',
    24: 'white_wolf',
    25: 'African_hunting_dog',
    26: 'Arctic_fox',
    27: 'lion',
    28: 'meerkat',
    29: 'ladybug',
    30: 'rhinoceros_beetle',
    31: 'ant',
    32: 'black-footed_ferret',
    33: 'three-toed_sloth',
    34: 'rock_beauty',
    35: 'aircraft_carrier',
    36: 'ashcan',
    37: 'barrel',
    38: 'beer_bottle',
    39: 'bookshop',
    40: 'cannon',
    41: 'carousel',
    42: 'carton',
    43: 'catamaran',
    44: 'chime',
    45: 'clog',
    46: 'cocktail_shaker',
    47: 'combination_lock',
    48: 'crate',
    49: 'cuirass',
    50: 'dishrag',
    51: 'dome',
    52: 'electric_guitar',
    53: 'file',
    54: 'fire_screen',
    55: 'frying_pan',
    56: 'garbage_truck',
    57: 'hair_slide',
    58: 'holster',
    59: 'horizontal_bar',
    60: 'hourglass',
    61: 'iPod',
    62: 'lipstick',
    63: 'miniskirt',
    64: 'missile',
    65: 'mixing_bowl',
    66: 'oboe',
    67: 'organ',
    68: 'parallel_bars',
    69: 'pencil_box',
    70: 'photocopier',
    71: 'poncho',
    72: 'prayer_rug',
    73: 'reel',
    74: 'school_bus',
    75: 'scoreboard',
    76: 'slot',
    77: 'snorkel',
    78: 'solar_dish',
    79: 'spider_web',
    80: 'stage',
    81: 'tank',
    82: 'theater_curtain',
    83: 'tile_roof',
    84: 'tobacco_shop',
    85: 'unicycle',
    86: 'upright',
    87: 'vase',
    88: 'wok',
    89: 'worm_fence',
    90: 'yawl',
    91: 'street_sign',
    92: 'consomme',
    93: 'trifle',
    94: 'hotdog',
    95: 'orange',
    96: 'cliff',
    97: 'coral_reef',
    98: 'bolete',
    99: 'ear'
}

with open("F:/DLdata/label_100.json", "w") as f:
    json.dump(label_class, f)

with open("F:/DLdata/label.json") as f:
    label = json.load(f)


def getCAM(img, layer_name, weights, cls_idx):
    cls_weights = weights[cls_idx].detach().unsqueeze(0)

    feature_extractor = create_feature_extractor(model, return_nodes={layer_name: "feature_map"})
    forward = feature_extractor(img)

    b, c, h, w = forward["feature_map"].shape
    feature_map = forward["feature_map"].detach().reshape(c, h * w)

    CAM = torch.mm(cls_weights, feature_map).reshape(h, w)
    CAM = (CAM - torch.min(CAM)) / (torch.max(CAM) - torch.min(CAM))
    CAM = (CAM.numpy() * 255).astype("uint8")

    return CAM


transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# 构建模型
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model.eval()

# model = torchvision.models.vgg16(weights=None)
# del model.classifier
# model.avgpool.output_size = (1, 1)
# model.add_module("classifier", torch.nn.Linear(512, 100))
# model.load_state_dict(torch.load(r"F:\DLdata\vggcam.pth"))

model.eval()
# 全连接层的权重
last_layer = list(model.modules())[-1]
weights = last_layer.weight
print(weights.shape)
# "F:/DLdata/Mini-ImageNet/new_images/train/n04509417/n0450941700000001.jpg"
# "F:\DLdata\Mini-ImageNet\new_images\train\n03220513\n0322051300000603.jpg"
img_path = r"F:\DLdata\Mini-ImageNet\new_images\train\n03220513\n0322051300000603.jpg"
original_img = Image.open(img_path)
# W, H = original_img.size

# w是、\\\
img = transform(original_img).unsqueeze(0)
output = model(img)
psort = torch.sort(F.softmax(output, dim=1), descending=True)
prob, cls_idx = psort
# w
top5 = [(i.item(), j.item()) for i, j in zip(cls_idx.view(-1), prob.view(-1))][:5]

fig, axs = plt.subplots(2, 3)
axs.reshape(-1)[0].imshow(np.asarray(original_img))

for idx, cls_prob in enumerate(top5):
    CAM = getCAM(img, "layer4", weights, cls_prob[0])
    upsample = cv2.resize(CAM, original_img.size)

    heatmap = cv2.applyColorMap(upsample, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    result = heatmap * 0.6 + np.asarray(original_img) * 0.4

    axs.reshape(-1)[idx + 1].imshow(np.uint8(result))
    axs.reshape(-1)[idx + 1].text(-10, -10, f"{label[str(cls_prob[0])][1]}: {cls_prob[1]:.3f}", fontsize=12,
                                  color="black")
    # axs.reshape(-1)[idx + 1].text(-10, -10, f"{label[str(cls_prob[0])]}: {cls_prob[1]:.3f}", fontsize=12,
    #                               color="black")
plt.savefig("cam.jpg", dpi=300, bbox_inches="tight")
plt.show()
# import yaml
# 快速排序
# 这是一个
# 这是一个服务器，
# import yaml
import yaml
# import yaml
# import
# import yaml
# with open(r"D:\Pycharm\yolov5-master\models\yolov5s.yaml") as f:
#     cfg = yaml.safe_load(f)
# print(cfg["one"])

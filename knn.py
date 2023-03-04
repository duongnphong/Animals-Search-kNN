from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from infer import transform
from PIL import Image
from model import model_eval
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def data(datadir):
    im_list = []
    npy_list = []
    label_list = []
    for class_names in tqdm(os.listdir(datadir)):
        dirpath = os.path.join(datadir, class_names)
        for i in os.listdir(dirpath):
            impath = os.path.join(dirpath, i)
            im_list.append(impath)
            extension = impath.split(".")[-1]
            npy_name = impath.replace("data", "feature").replace(extension, "npy")
            npy_list.append(np.load(npy_name).squeeze(0))
            label_list.append(class_names)
    return im_list, npy_list, label_list



knn_model = KNeighborsClassifier(n_neighbors=20, metric="cosine", weights="distance")

im_list, X_train, label = data("data/")



label_encoder = preprocessing.LabelEncoder()
y_train = label_encoder.fit_transform(label)
# print(X_train[0].shape)

knn_model.fit(X_train, y_train)


def knn_infer(imgpath):
    model = model_eval().to("cpu")
    im = Image.open(imgpath).convert("RGB")
    im = transform(im).unsqueeze(0)
    feature = model(im)
    feature = feature.detach().numpy()
    return feature.squeeze(0)

impath = "/home/phong/Desktop/why-are-there-so-many-1.jpg"
feature = knn_infer(impath)
# feature = np.load('feature/elephant/e83cb00a2ef1053ed1584d05fb1d4e9fe777ead218ac104497f5c978a4efbcb0_640.npy').squeeze(0)

distance = knn_model.kneighbors([feature])
pred = knn_model.predict([feature])
prob = knn_model.predict_proba([feature])
# print(distance[1][0])

# im_list = append_list("data/")
query = []
distances = []
for i in distance[1][0]:
    query.append(im_list[i])
for i in distance[0][0]:
    distances.append(i)



plt.figure(figsize=(15, 15))
plt.subplot(5, 5, 3)
plt.axis(False)
img0 = np.asarray(Image.open(impath))
plt.imshow(img0)
for i in range(20):
    img = f"img{i+6}"
    plt.subplot(5, 5, i+6)
    plt.axis(False)
    plt.title(f"Distance: {distances[i]:.4f}")
    img = np.asarray(Image.open(query[i]))
    plt.imshow(img)

plt.savefig('test.jpg')
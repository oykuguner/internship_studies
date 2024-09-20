from fastc import Fastc, Kernels, Pooling, ModelTemplates, Template

# Basit veri seti
tuples = [
    ("I am very happy with the results.", 'positive'),
    ("This is a very disappointing outcome.", 'negative'),
    ("I am excited about the new project!", 'positive'),
    ("I am not happy with the recent changes.", 'negative'),
    ("I had a great time with my friends at the party.", 'positive'),
    ("My vacation was wonderful and relaxing.", 'positive'),
    ("I didn't get any sleep last night because of the noise.", 'negative'),
    ("The food at the restaurant was absolutely delicious.", 'positive'),
    ("I missed my flight because of the traffic jam.", 'negative'),
    ("I feel refreshed after a great workout at the gym.", 'positive'),
    ("My car broke down in the middle of nowhere.", 'negative'),
    ("I enjoyed the movie; it was very entertaining.", 'positive'),
    ("I didn't like the presentation; it was boring.", 'negative'),
]

# Modeli En Yakın Merkez (Nearest Centroid) ile Eğitme
model_nearest_centroid = Fastc(
    embeddings_model='microsoft/deberta-base',
    kernel=Kernels.NEAREST_CENTROID,
)

model_nearest_centroid.load_dataset(tuples)
model_nearest_centroid.train()

# Modeli Lojistik Regresyon ile Eğitme
model_logistic_regression = Fastc(
    embeddings_model='microsoft/deberta-base',
    kernel=Kernels.LOGISTIC_REGRESSION,
    cross_validation_splits=2,
)

model_logistic_regression.load_dataset(tuples)
model_logistic_regression.train()

# Modeli Pooling ile Eğitme
model_pooling = Fastc(
    embeddings_model='microsoft/deberta-base',
    pooling=Pooling.MEAN_MASKED,
)

model_pooling.load_dataset(tuples)
model_pooling.train()

# Modeli Şablon ile Eğitme ve Kaydetme
template_text = ModelTemplates.E5_INSTRUCT

model_template = Fastc(
    embeddings_model='intfloat/multilingual-e5-large-instruct',
    template=Template(
        ModelTemplates.E5_INSTRUCT,
        instruction='Classify as positive or negative'
    ),
)

model_template.load_dataset(tuples)
model_template.train()

model_template.save_model('./sentiment-classifier/')

# Yerel Olarak Kaydedilen Modeli Yükleme ve Tahmin Yapma
loaded_model = Fastc('./sentiment-classifier/')

sentences = [
    'I am feeling well.',
    'I am in pain.',
]

# Tek bir tahmin
print("Single Prediction:")
scores_single = loaded_model.predict_one(sentences[0])
print(f"Sentence: {sentences[0]}")
print(f"Predicted Label: {scores_single['label']}")

# Toplu tahmin
print("\nBatch Predictions:")
scores_batch = loaded_model.predict(sentences)
for sentence, scores in zip(sentences, scores_batch):
    print(f"Sentence: {sentence}")
    print(f"Predicted Label: {scores['label']}")

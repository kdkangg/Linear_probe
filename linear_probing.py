from transformers import AutoFeatureExtractor, BeitForMaskedImageModeling
from datasets import load_dataset
import evaluate
from sklearn.linear_model import LogisticRegression
import torch
from tqdm import tqdm

train = load_dataset('cifar10', split= 'train').with_format("torch")
test = load_dataset('cifar10', split='test').with_format("torch")


train_imgs = [img for img in train['img']]
test_imgs = [img for img in test['img']]
train_labels = [label for label in train['label']]
test_labels = [label for label in test['label']]

feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k')
beit = BeitForMaskedImageModeling.from_pretrained('microsoft/beit-base-patch16-224-pt22k')

beit.eval()

n_beit_layers = len(beit.beit.encoder.layer)

#extract hidden states
def beit_output(data):
    features = feature_extractor(data, return_tensors='pt')
    output = beit(**features, output_hidden_states = True)
    return output

train_outputs = beit_output(train_imgs)
test_outputs = beit_output(test_imgs)

train_layers = []
test_layers = []

def extract_layer(list, outputs):
    #number of datapoints
    n = outputs.logits.shape[0]

    with torch.no_grad():
        for i in range(n_beit_layers):
            list.append(outputs.hidden_states[i].reshape(n,-1).cpu().detach().numpy())
    return list

extract_layer(train_layers, train_outputs)
extract_layer(test_layers, test_outputs)

#linear classification
model = LogisticRegression()
result = []

for i in tqdm(range(n_beit_layers)):
    model.fit(train_layers[i], train_labels)
    pred = model.predict(test_layers[i])

    metric = evaluate.load('accuracy')
    accuracy = metric.compute(predictions=pred, references=test_labels)

    result.append(accuracy.values())

for i in range(n_beit_layers):
    print(f'layer: {i}, accuracy : {result[i]}')
from transformers import AutoFeatureExtractor, BeitForMaskedImageModeling
from datasets import load_dataset
import evaluate
from sklearn.linear_model import LogisticRegression
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

train = load_dataset('cifar10', split= 'train').with_format("torch")
test = load_dataset('cifar10', split='test').with_format("torch")


train_imgs = [img for img in train['img']]
test_imgs = [img for img in test['img']]
train_labels = [label for label in train['label']]
test_labels = [label for label in test['label']]

feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k')
beit = BeitForMaskedImageModeling.from_pretrained('microsoft/beit-base-patch16-224-pt22k')

beit.to(device)
beit.eval()

n_beit_layers = len(beit.beit.encoder.layer)

#feature extract
train_features = feature_extractor(train_imgs[:1000], return_tensors = 'pt').to(device)
test_features = feature_extractor(test_imgs[:100], return_tensors = 'pt').to(device)

train_outputs = beit(**train_features, output_hidden_states = True)
test_outputs = beit(**test_features, output_hidden_states = True)

train_layers = []
test_layers = []

def extract_layer(list, outputs):
    #number of datapoints
    n = outputs.logits.shape[0]

    with torch.no_grad():
        for i in range(n_beit_layers):
            list.append(outputs.hidden_states[i].reshape(n,-1).cpu().detach().numpy()) #instead of hidden state, norm layer?? -> check
    return list

extract_layer(train_layers, train_outputs)
extract_layer(test_layers, test_outputs)

#linear classification
model = LogisticRegression()

result = []

for i in range(n_beit_layers):
    model.fit(train_layers[i],train_labels[:1000])
    pred = model.predict(test_layers[i])

    metric = evaluate.load('accuracy')
    accuracy = metric.compute(predictions=pred, references=test_labels[:100])
    
    result.append(accuracy.values())
    print(f'layer: {i}, accuracy : {accuracy.values()}')
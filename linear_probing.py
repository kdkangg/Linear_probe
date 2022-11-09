from transformers import AutoFeatureExtractor, BeitForMaskedImageModeling
from datasets import load_dataset
import evaluate
from sklearn.linear_model import LogisticRegression
import torch
from tqdm import tqdm

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

train = load_dataset('cifar10', split='train').with_format("torch")
test = load_dataset('cifar10', split='test').with_format("torch")

train_imgs = [img for img in train['img']]
test_imgs = [img for img in test['img']]
train_labels = [label for label in train['label']]
test_labels = [label for label in test['label']]

feature_extractor = AutoFeatureExtractor.from_pretrained(
    'microsoft/beit-base-patch16-224-pt22k')
beit = BeitForMaskedImageModeling.from_pretrained(
    'microsoft/beit-base-patch16-224-pt22k')

beit.to(device)
beit.eval()

n_beit_layers = len(beit.beit.encoder.layer)

#extract hidden states
def beit_output(data,device):
    features = feature_extractor(
        data, return_tensors='pt').to(device)
    output = beit(**features, output_hidden_states=True)
    return output


train_outputs = beit_output(train_imgs, device=device)
test_outputs = beit_output(test_imgs, device=device)

train_layers = []
test_layers = []


def extract_layer(list, outputs):
    #number of datapoints
    n = outputs.logits.shape[0]

    with torch.no_grad():
        for i in range(n_beit_layers):
            l_output = outputs.hidden_states[i][:, 1:, :].reshape(
                n, -1)#.cpu().detach().numpy()
            list.append(torch.mean(l_output,1))
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

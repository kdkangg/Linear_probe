from transformers import AutoFeatureExtractor, BeitForMaskedImageModeling
from datasets import load_dataset
import evaluate
from sklearn.linear_model import LogisticRegression
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load CIFAR 10 data
train = load_dataset('cifar10', split='train').with_format("torch")
test = load_dataset('cifar10', split='test').with_format("torch")
train_bn = DataLoader(train, batch_size=100)
test_bn = DataLoader(test,batch_size=100)

feature_extractor = AutoFeatureExtractor.from_pretrained(
    'microsoft/beit-base-patch16-224-pt22k')
beit = BeitForMaskedImageModeling.from_pretrained(
    'microsoft/beit-base-patch16-224-pt22k')
beit.to(device)
beit.eval()

n_beit_layers = len(beit.beit.encoder.layer) #number of beit layers


def beit_output(data_bn,device):
    outputs = []
    labels =[]
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_bn)):
            features = feature_extractor([i for i in data['img']], return_tensors='pt').to(device)
            output = beit(**features, output_hidden_states=True,return_dict=True)

            layer_output = []
            for i in range(n_beit_layers):
                l_output = output.hidden_states[i][:, 1:, :]

                layer_output.append(torch.mean(l_output,1))
                
            outputs.append(torch.stack(layer_output))
            labels.append(data['label'])

    return torch.stack(outputs), torch.cat(labels) 

train_outputs, train_labels = beit_output(train_bn, device) 
test_outputs, test_labels = beit_output(test_bn, device)


#linear classification
model = LogisticRegression(max_iter = 1000)
result = []

emb_dim = train_outputs.shape[-1]

for i in tqdm(range(n_beit_layers)):
    model.fit(train_outputs[:,i,:].cpu().detach().numpy().reshape(-1,emb_dim), train_labels)
    pred = model.predict(test_outputs[:,i,:].cpu().detach().numpy().reshape(-1,emb_dim))

    metric = evaluate.load('accuracy')
    accuracy = metric.compute(predictions=pred, references=test_labels)

    result.append(accuracy['accuracy'])

for i in range(n_beit_layers):
    print(f'layer: {i}, accuracy : {result[i]}')

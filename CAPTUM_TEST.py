# Test against Captum
!pip install captum


# Test against Captum

from captum.attr import Lime
import torch
import numpy as np

def calculate_similarity(explanation_implemented_lime, explanation_captum_lime):
    """
    Calculate similarity between two sets of explanations using Mean Absolute Error (MAE)
    """

    explanation_implemented_lime = {key.lower(): value for key, value in explanation_implemented_lime.items()}

    common_keys = set(explanation_implemented_lime.keys()) & set(explanation_captum_lime.keys())
    tensor_imp_lime = torch.tensor([explanation_implemented_lime[key] for key in common_keys])
    tensor_cap_lime = torch.tensor([explanation_captum_lime[key] for key in common_keys])

    mae = torch.abs(tensor_imp_lime - tensor_cap_lime).mean().item()
    return mae
# Test the setup with model certainty
corpus_test = [item[1] for item in test_dataset]
with open('exp_file_emb_var75.txt', 'r') as f:
    exp_file = [eval(i) for i in f.readlines()]
f.close()
sim = []
model = model.to(device)
for ind in range(len(corpus_test)):
    index = ind
    token_ids = text_transform(corpus_test[index])
    text_lis = [token_ids]
    length = [len(token_ids)]
    length_tensor = torch.tensor(length, dtype=torch.int64)
    padded_text = torchtext.functional.to_tensor(text_lis, padding_value=1).to(device)
    captum_lime = Lime(model)
    captum_lime_attr = captum_lime.attribute(padded_text,  n_samples=200)
    lime_words = [vocab.get_itos()[i] for i in token_ids]
    explanation_captum_lime = dict(zip(lime_words, captum_lime_attr[0].cpu().numpy()))
    explanation_captum_lime = dict(sorted(explanation_captum_lime.items(), key=lambda x: x[1], reverse=True))
    explanation_implemented_lime = exp_file[ind]
    # Calculate similarity using MAE
    similarity = calculate_similarity(explanation_implemented_lime, explanation_captum_lime)
    sim.append(similarity)
sim = np.array(sim)
print(sim.mean(), sim.var())

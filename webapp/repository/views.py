import os, sys
from typing import Any, Dict
from django.shortcuts import render, redirect ; # type: ignore
from django.http import HttpRequest, JsonResponse ; # type: ignore
from django.views.generic import TemplateView ; # type: ignore
import torch
import numpy as np
import glob
from transformers import GPT2TokenizerFast
sys.path.append(os.path.abspath(os.path.pardir))
import models.PretrainedModels.GPT2.GPT2Models as gpt2
import models.NewModel.Transformer.TransformerModel as transformer
from transformers import pipeline, set_seed

# global generator

# generator = pipeline('text-generation', model='gpt2')
set_seed(42)


def about_view(request: HttpRequest):
    """ About page view for the website

    Args:
        request (HttpRequest): The request object
    
    Returns:
        HttpResponse: The response object
    """
    return render(request, 'repository/about.html')


# def contact_view(request: HttpRequest):
#     """ Contact page view for the website

#     Args:
#         request (HttpRequest): The request object
    
#     Returns:
#         HttpResponse: The response object
#     """
#     return render(request, 'repository/contact.html')


def faq_view(request: HttpRequest):
    """ FAQ page view for the website

    Args:
        request (HttpRequest): The request object
    
    Returns:
        HttpResponse: The response object
    """
    return render(request, 'repository/faq.html')


class SearchView(TemplateView):
    """ The search view is where the classification of the news to be Fake/True.
    """
    redirect_field_name = 'redirect_to'
    template_name = 'repository/main.html'
    success_url = '/'
    
    def get_context_data(self, **kwargs: Any) -> Dict[str, Any]:
        context = super().get_context_data(**kwargs)
        print(self.kwargs)
        if self.kwargs.get('results') == "1":
            context["results"] = "True"
        elif self.kwargs.get('results') == "0":
            context["results"] = "False"
        # context["accuracy"] = self.kwargs.get('accuracy')
        return context

    def get(self, request: HttpRequest, *args: Any, **kwargs: Any) -> Any:
        return render(request, self.template_name, {'form': None})
 
    

def search_view(request: HttpRequest):
    """ Search if the input text is fake or not.

    Args:
        request (HttpRequest): Request input.

    Returns:
        JsonResponse: Response for the input data to be Fake/True.
    """
    queryDict = request.GET
    text = queryDict.get('search')
 
    print("Input news:", text)
    print("loading the model")
    
    # model_dir = os.path.abspath(os.getcwd() +"/../models/NewModel/Transformer/model_trans.pth")
    model = transformer.model
    tokenizer = transformer.tokenizer
    # model.load_state_dict(torch.load(model_dir))
    # print("model is loaded!")
        
    # tokenize and encode sequences in the test set
    MAX_LENGHT = model.max_length
    tokens_unseen = tokenizer.batch_encode_plus(
        [text],
        max_length = MAX_LENGHT,
        pad_to_max_length=True,
        truncation=True
    )

    unseen_seq = torch.tensor(tokens_unseen['input_ids'])
    unseen_mask = torch.tensor(tokens_unseen['attention_mask']).to(transformer.device)

    with torch.no_grad():
        # unseen_seq = unseen_seq.to(transformer.device)
        preds = model(unseen_seq)
        preds = preds.detach().cpu().numpy()
    result = "Fake" if np.mean(preds) > 1 else "True"
    print(result)
    return JsonResponse(data={'response': result})


def autoComplete(request: HttpRequest):
    """ This function is used to get the auto complete text from the model"""
    global generator
    input_text = request.GET.get('search', '')
    
    # generator = pipeline('text-generation', model='gpt2')
    response = generator(input_text, max_length=50, num_return_sequences=1, repetition_penalty=1.5, top_k=50, top_p=0.95, temperature=1.0)
    
    # Put in json
    response = response[0]['generated_text']
    responseData = {'response': response}
    return JsonResponse(data=responseData, safe=False)
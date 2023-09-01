import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import math
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AdamW
from tqdm import tqdm_notebook


def train(model, dataloader, optimizer, criterion, scheduler=None, device='cpu'):
    model.train()

    # Record total loss
    total_loss = 0.

    # Get the progress bar for later modification
    progress_bar = tqdm_notebook(dataloader, ascii=True)

    # Mini-batch training
    for batch_idx, data in enumerate(progress_bar):
        source = data[0].to(device)
        mask = data[1].to(device)
        target = data[2].unsqueeze(1).to(device)
        # target = data[2].to(device)
        
        if model.__class__.__name__ == 'FullTransformerTranslator':
            translation = model(source, target)
        else:
            translation = model(source, mask)
        optimizer.zero_grad()
        translation = translation.mean(1)
        target = target.float()
        
        loss = criterion(translation, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_description_str(
            "Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    return total_loss, total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device='cpu'):
    # Set the model to eval mode to avoid weights update
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        # Get the progress bar
        progress_bar = tqdm_notebook(dataloader, ascii=True)
        for batch_idx, data in enumerate(progress_bar):
            source = data[0].to(device)
            mask = data[1].to(device)
            target = data[2].unsqueeze(1).to(device)
            # target = data[2].to(device)

            if model.__class__.__name__ == 'FullTransformerTranslator':
                translation = model(source, target)
            else:
                translation = model(source, mask)
            translation = translation.mean(1)
            target = target.float()
            loss = criterion(translation, target)
            total_loss += loss.item()
            progress_bar.set_description_str(
                "Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    avg_loss = total_loss / len(dataloader)
    return total_loss, avg_loss


def find_accuracy(model, dataloader, device, batch_size):
    # Set the model to eval mode to avoid weights update
    model.eval()
    correct_pred = 0
    with torch.no_grad():
        # Get the progress bar
        progress_bar = tqdm_notebook(dataloader, ascii=True)
        for batch_idx, data in enumerate(progress_bar):
            source = data[0].to(device)
            mask = data[1].to(device)
            target = data[2].unsqueeze(1).to(device)
            # target = data[2].to(device)

            if model.__class__.__name__ == 'FullTransformerTranslator':
                translation = model(source, target)
            else:
                translation = model(source, mask)
            translation = translation.mean(1)
            pred = (translation > 0).int()
            target = target.float()
            correct_items = ((pred == target).int()).sum().item()
            correct_pred += correct_items
            progress_bar.set_description_str(
                "Batch: %d, Correct items: %.4f" % ((batch_idx + 1), correct_items))
    accuracy = correct_pred / (len(dataloader)*batch_size)
    return accuracy


# def train(model, train_dataloader, optimizer, criterion, device):
#     print("\nTraining...")
#     model.train()
#     total_loss, total_accuracy = 0, 0

#     for step,batch in enumerate(train_dataloader):
#         if step % 50 == 0 and not step == 0:
#             print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
#         batch = [r for r in batch]
#         sent_id, mask, labels = batch
#         model.zero_grad()
#         preds = model(sent_id, mask)
#         loss = criterion(preds, labels)
#         total_loss = total_loss + loss.item()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#         preds=preds.detach().cpu().numpy()

#     avg_loss = total_loss / len(train_dataloader)

#     return total_loss, avg_loss

# def evaluate(model, val_dataloader, criterion, device):
#     print("\nEvaluating...")
#     model.eval()
#     total_loss, total_accuracy = 0, 0
#     for step,batch in enumerate(val_dataloader):
#         if step % 50 == 0 and not step == 0:
#             print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))
#         batch = [t for t in batch]
#         sent_id, mask, labels = batch
#         with torch.no_grad():
#             preds = model(sent_id, mask)
#             loss = criterion(preds,labels)
#             total_loss = total_loss + loss.item()
#             preds = preds.detach().cpu().numpy()
#     avg_loss = total_loss / len(val_dataloader)
#     return total_loss, avg_loss


# def find_accuracy(model:nn.Module, dataloader:list, device:str, batch_size:int) -> float:
#     """ Get the accuracy of the model on the given dataset.

#     Args:
#         model (nn.Module): model to be used for prediction.
#         dataset (list): dataset to be used for prediction.
#         device (str): device to be used for training options: 'cuda' or 'cpu'
#         batch_size (int): batch size to be used for training.

#     Returns:
#         float: Accuracy of the model on the given dataset.
#     """
#     # Set the model to eval mode to avoid weights update
#     model.eval()
#     correct_pred = 0
#     with torch.no_grad():
#         for step,batch in enumerate(dataloader):
#             batch = [t for t in batch]
#             sent_id, mask, labels = batch
#             translation = model(sent_id, mask)
#             translation = translation.mean(1)
#             pred = (translation > 0).int()
#             labels = labels.float()
#             correct_items = ((pred == labels).int()).sum().item()
#             correct_pred += correct_items
#     accuracy = correct_pred / (len(dataloader)*batch_size)
#     return accuracy


def benchmark_1(cwd:str, model:object, name:str, tokenizer:object, device:str, learning_rate:float=0.0001):
    """ Benchmarking the model by running training and validation
        for 10 epochs and generate a standardize report and images for learning curves.
        
    Args:
        cwd (str): current working directory. Please set this according to your working directory for the notebook.
        model (object): model to be trained
        name (str): name to associate to current model benchmark results.
        tokenizer (object): tokenizer to be used for tokenizing the text.
        device (str): device to be used for training options: 'cuda' or 'cpu'
        learning_rate (float): learning rate to be used for training. Default: 0.0001
    """
    #### Get the data ####
    print('Loading data...')
    print("os.getcwd():", os.getcwd())
    databaseDir = os.path.join(os.getcwd(), './Data')
    true_data = pd.read_csv(databaseDir+'/True_original.csv')
    fake_data = pd.read_csv(databaseDir+'/True_original.csv')
    
    # Only for testing
    # true_data = true_data[:100]
    # fake_data = fake_data[:100]

    true_data['Target'] = ['True']*len(true_data)
    fake_data['Target'] = ['Fake']*len(fake_data)

    data = true_data.append(fake_data).sample(frac=1).reset_index().drop(columns=['index'])

    data['label'] = pd.get_dummies(data.Target)['Fake']

    train_text, temp_text, train_labels, temp_labels = train_test_split(data['title'], data['label'],
                                                                        random_state=2018,
                                                                        test_size=0.3,
                                                                        stratify=data['Target'])
    val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                    random_state=2018,
                                                                test_size=0.5,
                                                                stratify=temp_labels)    
    benchmark_inner(cwd, data, name, model,  learning_rate, tokenizer, train_text, train_labels, val_text,\
    val_labels, test_text, test_labels, )


def benchmark_isot(cwd:str, model:object, name:str, tokenizer:object, device:str, learning_rate:float=0.0001, epochs=5):
    """ Benchmarking the model by running training and validation
        for 10 epochs and generate a standardize report and images for learning curves.
        
    Args:
        cwd (str): current working directory. Please set this according to your working directory for the notebook.
        model (object): model to be trained
        name (str): name to associate to current model benchmark results.
        tokenizer (object): tokenizer to be used for tokenizing the text.
        device (str): device to be used for training options: 'cuda' or 'cpu'
        learning_rate (float): learning rate to be used for training. Default: 0.0001
        epochs (int): number of epochs to be used for training. Default: 10
    """
    #### Get the data ####
    print('Loading data...')
    print("os.getcwd():", os.getcwd())

    data = pd.read_csv(os.getcwd()+'./Data/final_fake_news.csv', delimiter=';')
    data['Target'] = 'True'
    data.loc[data['label'] == 0, 'Target'] = 'Fake'
    data = data.sample(frac=1).reset_index().drop(columns=['index'])
    data['label'] = pd.get_dummies(data.Target)['Fake']

    train_text, temp_text, train_labels, temp_labels = train_test_split(data['text'], data['label'],
                                                                        random_state=2018,
                                                                        test_size=0.3,
                                                                        stratify=data['Target'])
    val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                    random_state=2018,
                                                                test_size=0.5,
                                                                stratify=temp_labels)    
    
    #### Generate Benchmark Results Directory ####
    benchmark_inner(cwd, data, name, model,  learning_rate, tokenizer, train_text, train_labels, val_text,\
    val_labels, test_text, test_labels, epochs=epochs)
    return


def benchmark_inner(cwd:str, data, name, model, learning_rate, tokenizer, train_text, train_labels, val_text,\
    val_labels, test_text, test_labels, batch_size = 32, max_length=15, epochs=10):
    #### Generate Benchmark Results Directory ####
    outputdir = os.path.join(cwd, 'BenchmarkResults', name)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    outputdir += '/'
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr = learning_rate)
    # criterion  = nn.NLLLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    model_path = outputdir+'model.pth'

    tokens_train = tokenizer.batch_encode_plus(
        train_text.tolist(),
        max_length = max_length,
        pad_to_max_length=True,
        truncation=True
    )
    tokens_val = tokenizer.batch_encode_plus(
        val_text.tolist(),
        max_length = max_length,
        pad_to_max_length=True,
        truncation=True
    )
    tokens_test = tokenizer.batch_encode_plus(
        test_text.tolist(),
        max_length = max_length,
        pad_to_max_length=True,
        truncation=True
    )
    
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_labels.tolist())

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    val_y = torch.tensor(val_labels.tolist())

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    test_y = torch.tensor(test_labels.tolist())

    train_data = TensorDataset(train_seq, train_mask, train_y)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_seq, val_mask, val_y)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

    test_data = TensorDataset(test_seq, test_mask, test_y)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size=batch_size)
    
    best_train_loss = float('inf')
    best_valid_loss = float('inf')
    best_train_perplexity = float('inf')
    best_valid_perplexity = float('inf')
    best_train_acc = float('inf')
    best_valid_acc = float('inf')
    train_losses=[]
    valid_losses=[]
    train_per = []
    valid_per = []
    train_acc_list = []
    valid_acc_list = []

    fp = open(outputdir+"benchmark.csv", "w")
    fp.write("epoch,train_loss, val_loss, train_perplexity, val_perplexity, train_acc, val_acc\n")
    
    for epoch_idx in range(epochs):
        print("-----------------------------------")
        print("Epoch %d" % (epoch_idx+1))
        print("-----------------------------------")

        train_loss, avg_train_loss = train(
            model, train_dataloader, optimizer, criterion, device=device)
        scheduler.step(train_loss)

        val_loss, avg_val_loss = evaluate(
            model, val_dataloader, criterion, device=device)

        train_perplexity = np.exp(avg_train_loss)
        val_perplexity = np.exp(avg_val_loss)
        
        acc_train = find_accuracy(model, train_dataloader, device=device, batch_size=batch_size)
        acc_val = find_accuracy(model, val_dataloader, device=device, batch_size=batch_size)
        print("Training Loss: %.4f. Validation Loss: %.4f. " %
            (avg_train_loss, avg_val_loss))
        print("Training Perplexity: %.4f. Validation Perplexity: %.4f. " %
            (np.exp(avg_train_loss), np.exp(avg_val_loss)))
        print("Training Acc: %.4f. Validation Acc: %.4f. " %
            (acc_train, acc_val))

        if np.exp(avg_val_loss) < best_valid_loss:
            torch.save(model.state_dict(), model_path)
            best_train_loss = avg_train_loss
            best_valid_loss = avg_val_loss
            best_train_perplexity = train_perplexity
            best_valid_perplexity = val_perplexity
            best_train_acc = acc_train
            best_valid_acc = acc_val
            print("Saved model")
            
        train_per.append(train_perplexity)
        valid_per.append(val_perplexity)

        train_acc_list.append(acc_train)
        valid_acc_list.append(acc_val)
        
        train_losses.append(avg_train_loss)
        valid_losses.append(avg_val_loss)
                
        fp.write(str(epoch_idx)+","+str(train_loss)+","+str(val_loss)+","+str(train_perplexity)+","+str(val_perplexity)+","+str(acc_train)+","+str(acc_val)+"\n")

    fp.write("OverallBest"+","+str(best_train_loss)+","+str(best_valid_loss)+","+str(best_train_perplexity)+","+str(best_valid_perplexity)+","+str(best_train_acc)+","+str(best_valid_acc)+"\n")
    fp.close()
    
    plot(epochs, train_per, valid_per, train_acc_list, valid_acc_list, outputdir)
    
    # for epoch in range(epochs):
    #     print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

    #     train_loss, avg_train_loss = train(
    #         model, train_dataloader, optimizer, criterion, device=device)
    #     scheduler.step(train_loss)

    #     val_loss, avg_val_loss = evaluate(
    #         model, val_dataloader, criterion, device=device)

    #     train_perplexity = np.exp(avg_train_loss)
    #     val_perplexity = np.exp(avg_val_loss)
        
    #     acc_train = find_accuracy(model, train_dataloader, device=device, batch_size=batch_size)
    #     acc_val = find_accuracy(model, val_dataloader, device=device, batch_size=batch_size)
        
    #     print("Training Loss: %.4f. Validation Loss: %.4f. " %
    #         (avg_train_loss, avg_val_loss))
    #     print("Training Perplexity: %.4f. Validation Perplexity: %.4f. " %
    #         (np.exp(avg_train_loss), np.exp(avg_val_loss)))
    #     print("Training Acc: %.4f. Validation Acc: %.4f. " %
    #         (acc_train, acc_val))

    #     if np.exp(avg_val_loss) < best_valid_loss:
    #         torch.save(model.state_dict(), model_path)
    #         best_train_loss = avg_train_loss
    #         best_valid_loss = avg_val_loss
    #         best_train_perplexity = train_perplexity
    #         best_valid_perplexity = val_perplexity
    #         best_train_acc = acc_train
    #         best_valid_acc = acc_val
    #         print("Saved model")

    #     train_per.append(train_perplexity)
    #     valid_per.append(val_perplexity)

    #     train_acc_list.append(acc_train)
    #     valid_acc_list.append(acc_val)
        
    #     train_losses.append(avg_train_loss)
    #     valid_losses.append(avg_val_loss)        
    #     fp.write(str(epoch)+","+str(train_loss)+","+str(val_loss)+","+str(train_perplexity)+","+str(val_perplexity)+","+str(acc_train)+","+str(acc_val)+"\n")

    # fp.write("OverallBest"+","+str(best_train_loss)+","+str(best_valid_loss)+","+str(best_train_perplexity)+","+str(best_valid_perplexity)+","+str(best_train_acc)+","+str(best_valid_acc)+"\n")
    # fp.close()
    
    # plot(epochs, train_per, valid_per, train_acc_list, valid_acc_list, outputdir)
    return


def plot(epochs, train_per, valid_per, train_acc_list, valid_acc_list, outputdir):
    #### Plot train and validation loss ####
    x_range = range(1, epochs+1)
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121)
    ax1.plot(x_range, train_per, label='train_perplexity')
    ax1.plot(x_range, valid_per, label='validation_perplexity')
    ax1.set_title('Perplexity vs Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Perplexity')
    ax1.grid(True)
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.plot(x_range, train_acc_list, label='train_accuracy')
    ax2.plot(x_range, valid_acc_list, label='validation_accuracy')
    ax2.set_title('Accuracy vs Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(outputdir+'perplexity_acc_curves.png')



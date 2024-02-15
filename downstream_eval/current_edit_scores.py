###This file measures the efficacy of making the current edit given a larger number of edits has been made in the past


import os
import json
import math
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    average_array = []

    for i in range(len(data)):
        window = data[max(i - window_size + 1, 0) : i + 1 ]
        window_avg = sum(window) / len(window)
        average_array.append(window_avg)

    return average_array


if __name__ == '__main__':
    metrics = {
        'rewrite_prompts_probs' : [], 
        'paraphrase_prompts_probs' : [], 
        'neighborhood_prompts_probs' : []
    }

    algo = 'ROME'
    run = 'run_011'
    sample_num = '5'
    save_location = 'downstream_eval/' + algo + '_' + run + '/'
    os.makedirs(save_location, exist_ok=True)
    data_location = 'results/' + algo + '/' + run + '/'
    bucket_size = 5
    window_size = 10

    #get order of edits
    indices_filename = 'counterfact_sampled_unique_10_20391.json'
    f = open(indices_filename)
    sampled_indices = json.load(f)


    for e, element_index in enumerate(sampled_indices[sample_num]):
        
        filename = 'case_{}.json'.format(str(element_index))
        
        file_loc = data_location + filename
        
        if not os.path.exists(file_loc):
            break

        with open(file_loc, "r") as f:
            data = json.load(f)

        for metric in metrics:
            if metric in ['rewrite_prompts_probs', 'paraphrase_prompts_probs']:
                success = [element['target_new'] < element['target_true'] for element in data['post'][metric]]
            else:
                success = [element['target_new'] > element['target_true'] for element in data['post'][metric]]

            value = sum(success)/len(success)
            metrics[metric].append(value)

    #making individual bar plots
    for metric in metrics:
        x, y = [], []
        for i in range( math.ceil(len(metrics[metric])//bucket_size) ):
            x.append(i)

            start_index = i * bucket_size
            end_index = min((i + 1) * bucket_size, len(metrics[metric]))
            y.append(sum(metrics[metric][start_index: end_index]))

        plt.bar(x, y)
        plt.savefig(save_location + algo + '_score_' + metric + '.png')
        plt.close()


    metric_colors = {
        'rewrite_prompts_probs' : 'k', 
        'paraphrase_prompts_probs' : 'b', 
        'neighborhood_prompts_probs' : 'r'
    }
    metric_labels = {
        'rewrite_prompts_probs' : 'Efficacy Score', 
        'paraphrase_prompts_probs' : 'Paraphrase Score', 
        'neighborhood_prompts_probs' : 'Neighborhood Score'
    }

    #making overall plot
    plt.figure(figsize=(6.5, 6))
    for metric in metrics:
        x, y = [], []
        for i in range( math.ceil(len(metrics[metric])//bucket_size) ):
            x.append(i * bucket_size)

            start_index = i * bucket_size
            end_index = min((i + 1) * bucket_size, len(metrics[metric]))

            y.append((sum(metrics[metric][start_index: end_index]) / bucket_size) * 100)

        y_avg = moving_average(y, window_size)

        plt.plot(x, y, linestyle = '--', color = metric_colors[metric], linewidth = 0.2)
        plt.plot(x, y_avg, color = metric_colors[metric], label = metric_labels[metric], linewidth = 2)


    plt.legend(loc='upper left', bbox_to_anchor=(0.45, 1.28), ncol=1, fontsize=14)
    plt.xlabel('Number of Edits', fontsize=20)
    plt.ylabel('Edit Accuracy', fontsize=20)
    plt.xlim(0, 1200)
    plt.title(f"Sample {sample_num}", loc='left')
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.tight_layout()
    plt.savefig(save_location + algo + '_editing_score.png')
    plt.close()

    
    

        

        


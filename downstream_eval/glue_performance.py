import os
import json
import matplotlib.pyplot as plt

if __name__ == '__main__':
    metric_names = ['correct', 'f1', 'mcc', 'invalid']
    task_names = ['sst', 'mrpc', 'cola', 'rte']
    
    glue_eval = {'distance':{}}
    for task in task_names:
        glue_eval[task] = {}
        for metric in metric_names:
            glue_eval[task][metric] = {}


    algo = 'ROME'
    run = 'run_011'
    save_location = 'downstream_eval/' + algo + '_' + run + '/'
    os.makedirs(save_location, exist_ok=True)
    data_location = 'results/' + algo + '/' + run + '/glue_eval/'


    for filename in os.listdir(data_location):
        file_loc = data_location + filename
        if 'glue' in filename:
            with open(file_loc, "r") as f:
                data = json.load(f)

            edit_num = data['edit_num']
            sample_num = data['sample_num']

            #plot distance data
            for layer in data['distance_from_original']:
                if layer not in glue_eval['distance']:
                    glue_eval['distance'][layer] = {}

                glue_eval['distance'][layer][edit_num] = data['distance_from_original'][layer]

            for task in task_names:
                if task in data:
                    for metric in metric_names:
                        glue_eval[task][metric][int(edit_num)] = data[task][metric]

    task_dict = {'sst':'SST2', 'mrpc':'MRPC', 'cola':'COLA', 'rte':'RTE'}
    
    task_colors = {'sst':'r', 'mrpc':'b', 'cola':'g', 'rte':'k'}
    #plot metrics individual with number of edits
    for metric in metric_names:
        plt.figure(figsize=(6.5, 4.5))
        for task in task_names:
            sorted_dict = sorted(glue_eval[task][metric].items(), key=lambda item: item[0])

            x, y = [], []
            for edit_num, correct in sorted_dict:
                x.append(edit_num)
                if metric in ['f1', 'accuracy']:
                    y.append(correct * 100)
                else:
                    y.append(correct / 200)

            plt.plot(x,y, label = task_dict[task], linewidth =3, color=task_colors[task])

        plt.legend(fontsize=14)
        plt.xlabel('Number of Edits', fontsize=20)
        if metric == 'correct':
            metric = 'accuracy'
        plt.ylabel(metric.upper(), fontsize=20)
        plt.tick_params(axis='x', labelsize=18)
        plt.tick_params(axis='y', labelsize=18)
        plt.title(f"Sample {sample_num}")
        plt.tight_layout()
        
        plt.savefig(save_location + algo + '_' + 'glue_' + metric + '.png')
        plt.close()


    #plot distance as a function of number of edits
    metric = 'distance'
    for l, layer in enumerate(glue_eval[metric]):
        sorted_dict = sorted(glue_eval[metric][layer].items(), key=lambda item: item[0])

        x, y = [], []
        for edit_num, correct in sorted_dict:
            x.append(edit_num)
            y.append(correct)

        if l == 0:
            plt.plot(x,y, linewidth =3, color = 'r', label = 'Layer ' + str(layer))
        else:
            plt.plot(x,y, linewidth =3, label = 'Layer ' + str(layer))
        
    plt.legend(fontsize=14)
    plt.xlabel('Number of Edits', fontsize=20)
    plt.ylabel('Normalized Distance', fontsize=20)
    plt.xlim(0, 1200)
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.tight_layout()

    plt.savefig(save_location + algo + '_' + 'distance.png')
    plt.close()
            
    
    #plot glue performance as a function of number of edits
#here we create 10 samples of size 500 to sample the counterfact dataset from to visualize model editing effects

from dsets.counterfact import CounterFactDataset
import random
import json

random.seed(37)

if __name__ == '__main__':

    num_samples = 10 #number of samples
    sample_size = 20391
    write_flag = True
    output_filename = 'counterfact_sampled_unique_' + str(num_samples) + '_'  + str(sample_size) + '.json'

    dataset = CounterFactDataset('data')

    ####select unique subjects
    subject2rel = []
    selected_indices = []
    for i in range(len(dataset)):
        item = dataset.__getitem__(i)
        relation = item['requested_rewrite']['relation_id'].lower()
        subject = item['requested_rewrite']['subject'].lower()

        input_text = subject #+ '-' + relation    ####only editing unique subjects

        if input_text not in subject2rel:
            subject2rel.append(input_text)
            selected_indices.append(i)

    sampled_indices = {}
    for n in range(num_samples):
        random.shuffle(selected_indices)
        sampled_indices[n] = selected_indices[:sample_size]

    if write_flag:
        json_object = json.dumps(sampled_indices, indent=4)
        with open(output_filename , "w") as outfile:
            outfile.write(json_object)

from dsets.counterfact import CounterFactDataset

dataset = CounterFactDataset('data')

subject2rel = []
selected_indices = []
for i in range(len(dataset)):
    item = dataset.__getitem__(i)
    relation = item['requested_rewrite']['relation_id'].lower()
    subject = item['requested_rewrite']['subject'].lower()

    input_text = subject #+ '-' + relation

    if input_text not in subject2rel:
        subject2rel.append(input_text)
        selected_indices.append(i)




print(len(subject2rel), len(selected_indices))


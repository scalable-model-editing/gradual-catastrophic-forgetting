from datasets import load_metric, load_dataset

class SSTEval():
    def __init__(self, model, tokenizer, eval_split = 'validation'):
        self.model = model
        self.tokenizer = tokenizer

        #initialize dataset
        dataset = load_dataset("glue", "sst2")
        self.eval_dataset = dataset[eval_split]

        self._initialize_prompts()


    def _initialize_prompts(self):
        self.few_shot_context = '''\
Review : excruciatingly unfunny and pitifully unromantic
Sentiment : negative

Review : rich veins of funny stuff in this movie
Sentiment : positive

Review : by far the worst movie of the year
Sentiment : negative

Review : fashioning an engrossing entertainment out
Sentiment : positive\n\n\
'''

        self.prefix_prompt = 'Review : '
        self.postfix_prompt = '\nSentiment :'


    def _get_answer(self, generated_text):
        answer_text = generated_text.split('Sentiment :')[-1].strip().strip()

        if 'positive' in answer_text:
            return 1
        elif 'negative' in answer_text:
            return 0

        return -1


    def evaluate(self, print_logs = False):
        correct = 0
        incorrect = 0
        invalid = 0
        for s, (sentence, label) in enumerate(zip(self.eval_dataset['sentence'], self.eval_dataset['label'])):
            print(sentence, label)

            input_prompt = self.few_shot_context + self.prefix_prompt + sentence + self.postfix_prompt
            input_prompt_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to('cuda')

            max_len = input_prompt_ids.shape[1] + 3

            output = self.model.generate(input_prompt_ids,max_length = max_len, do_sample = False)
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

            answer = self._get_answer(generated_text)

            if answer == -1:
                invalid += 1
            elif answer == label:
                correct += 1
            else:
                incorrect += 1

            if print_logs:
                print(generated_text)
                print(correct, incorrect, invalid, s+1)
                print('--'*50)

        return correct, incorrect, invalid, s+1

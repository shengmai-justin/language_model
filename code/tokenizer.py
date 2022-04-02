class Tokenizer():
    def __init__(self):
        super(Tokenizer, self).__init__()
        self.id_to_word = {
            0: "<bos>",
            1: "<pad>",
            2: "<eos>",
            3: "<unk>",
        }
        self.word_to_id = {
            "<bos>": 0,
            "<pad>": 1,
            "<eos>": 2,
            "<unk>": 3,
        }

    def build_vocab(self, dataset):
        counter = {}
        for sentence in dataset:
            words = sentence.split(" ")
            for word in words:
                if word is not "":
                    if word in counter:
                        counter[word] += 1
                    else:
                        counter[word] = 1
        counter = sorted(counter.items(), key=lambda item: item[1])[::-1]
        index = 4
        for word in counter:
            if word[0] not in self.word_to_id:
                self.id_to_word[index] = word[0]
                self.word_to_id[word[0]] = index
                index += 1

    def tokenize(self, input):
        result = []
        for sentence in input:
            words = sentence.split(" ") + ["<eos>"]
            word_seq = []
            for word in words:
                if word is not "":
                    word_seq.append(self.word_to_id[word])
            if word_seq:
                result.append(word_seq)

        return result


def main():
    tokenizer = Tokenizer()
    with open("/Users/chenshengmai/Desktop/Spring 2022/language_model/data/train") as file:
        dataset = file.read().split("\n")
    tokenizer.build_vocab(dataset)
    example = ["this is sample code", "for a project", ""]
    print(tokenizer.tokenize(example))


if __name__ == "__main__":
    main()

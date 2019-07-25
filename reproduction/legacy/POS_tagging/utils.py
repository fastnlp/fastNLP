import pickle


def load_embed(embed_path):
    embed_dict = {}
    with open(embed_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.split(" ")
            if len(tokens) <= 5:
                continue
            key = tokens[0]
            if len(key) == 1:
                value = [float(x) for x in tokens[1:]]
                embed_dict[key] = value
    return embed_dict


if __name__ == "__main__":
    embed_dict = load_embed("/home/zyfeng/data/small.txt")

    print(embed_dict.keys())

    with open("./char_tencent_embedding.pkl", "wb") as f:
        pickle.dump(embed_dict, f)
    print("finished")

class MyDataloader:
    def load(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        data = self.parse(lines)
        return data

    def parse(self, lines):
        """
            [
                [word], [pos], [head_index], [head_tag]
            ]
        """
        sample = []
        data = []
        for i, line in enumerate(lines):
            line = line.strip()
            if len(line) == 0 or i + 1 == len(lines):
                data.append(list(map(list, zip(*sample))))
                sample = []
            else:
                sample.append(line.split())
        if len(sample) > 0:
            data.append(list(map(list, zip(*sample))))
        return data



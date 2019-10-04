import glob

class Read_all_data:
    def __init__(self):
        self.all_data_file = open('train_data/all_data/' + 'all_data_file.txt', 'w')
        self.files = glob.glob('train_data/' + '*.txt')
        for file in self.files:
            with open(file, 'r') as f:
                self.all_data_file.write(f.read())
            print(f)
if __name__ == '__main__':
    all = Read_all_data()
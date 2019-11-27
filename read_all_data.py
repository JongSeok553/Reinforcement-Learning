import glob

class Read_all_data:
    def __init__(self):
        self.all_data_file = open('model_supervised/4action/' + '4action_xyhd_all.txt', 'w')
        self.files = glob.glob('model_supervised/4action/' + '*.txt')
        for file in self.files:
            with open(file, 'r') as f:
                self.all_data_file.write(f.read())
            print(f)
if __name__ == '__main__':
    all = Read_all_data()
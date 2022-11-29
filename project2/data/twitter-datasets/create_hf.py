import numpy as np

USE_FULL_DATASET = True
PROJECT_PATH = "./"

def read_file(file_name_label_tuple):
    fname, label = file_name_label_tuple
    tweets, labels = [], []
    with open(fname, 'r', encoding='utf-8') as f:
        tweets = f.readlines()
    
    labels = [label] * (len(tweets))

    return(tweets, labels)


def load_train_data():

    if USE_FULL_DATASET == True:
        X_train_neg_path = PROJECT_PATH + "train_neg_full.txt"
        X_train_pos_path = PROJECT_PATH + "train_pos_full.txt"
        
    else:
        X_train_neg_path = PROJECT_PATH + "train_neg.txt"
        X_train_pos_path = PROJECT_PATH + "train_pos.txt"
    
    tweets, labels = read_file((X_train_neg_path, 0))
    tweets = list(set(tweets))
    labels = labels[:len(tweets)]
    print("There are ", len(tweets), " negative tweets after removing the duplicates.")
    
    tweets_2, labels_2 = read_file((X_train_pos_path, 1))
    tweets_2 = list(set(tweets_2))
    labels_2 = labels_2[:len(tweets_2)]
    print("There are ", len(tweets_2), " positive tweets after removing the duplicates.")

    
    tweets += tweets_2
    tweets_2 = []
    del(tweets_2)
    labels += labels_2
    labels_2 = []
    del(labels_2)
    print(f"Loaded {len(tweets)} tweets!")

    tweets, labels = np.array(tweets), np.array(labels)
    print(tweets)


    return tweets, labels


def create_data_file(tweets, labels):
    
    with open("HF_data.txt", "wb") as f:
        for i in range(len(tweets)):
            # print(tweets[i])
            f.write(f"{labels[i]} \t {tweets[i]}".encode('utf-8'))

    
    

def main():
    tweets, labels = load_train_data()
    create_data_file(tweets, labels)
    #sentiment = Sentiment()

if __name__ == "__main__":
    main()
import model.Attribute
import model.CNN
import model.train
import data.load_data
import data.participle

if __name__ == "__main__":
    attribute = model.Attribute.Attribute(name="TextCNN")
    print("loading pretrain vector...")
    word_to_id = data.load_data.load_pretrain_vector(attribute)
    #word_to_id = data.participle.load_vocab_file(attribute)
    hyperparameter = model.Attribute.Hyperparameters(embedding="embedding_sogounews.npz")
    
    print("loading txt...")
    articles,label = data.participle.load_txt_data(attribute=attribute)
    print("loading dataset...")
    dataset = data.participle.load_dataset(articles,label,attribute,hyperparameter,word_to_id)
    data_len = len(dataset)
    delim = (data_len // 10 )* 9
    train_iter = data.load_data.build_iterator(dataset[:delim], attribute,hyperparameter)
    validate_iter = data.load_data.build_iterator(dataset[delim:], attribute,hyperparameter)


    print("training...")
    CNN_classifier = model.CNN.TextCNN(
        attribute=attribute, hyperparameter=hyperparameter).to(attribute.device)
    model.train.weights_init(CNN_classifier)
    model.train.train(CNN_classifier,attribute,hyperparameter,train_iter,validate_iter)

    

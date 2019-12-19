import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle
import os 
from torchvision import transforms
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image

torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def test(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225))])
    
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print(len(vocab))
    encoder = EncoderCNN(args.embed_size).eval()
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    image = load_image(args.image, transform)
    image_tensor = image.to(device)

    feature = encoder(image_tensor)
    sample_ids = decoder.sample(feature)
    sample_ids = sample_ids[0].cpu().numpy()

    sampled_caption = []
    for word_id in sample_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word=='<end>':
            break
    sentence = ' '.join(sampled_caption)

    print(sentence)
    image = Image.open(args.image)
    plt.imshow(np.asarray(image))

# 设置参数进行测试
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='/picture/football.jpg', help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='models/encoder-5.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-5.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl', help='path for vocabulary wrapper')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    config = parser.parse_args()
    test(config)

import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader
from build_vocab import Vocabulary
from model import EncoderCNN,DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

torch.cuda.set_device(1)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_main(args):
    # 创建模型模型的文件夹
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # 配置transform: 图像预处理、归一化
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    # 加载词汇封装文件
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # 新建数据加载器
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

    # 实例化编码器和解码器
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    
    # 社会损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # 训练模型
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            
            # 设置mini-batch数据集
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # 前向传播-》反向传播-》优化
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            
            decoder.zero_grad() # 切记
            encoder.zero_grad() #切记
            loss.backward()
            
            optimizer.step()

            # 打印Log信息
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                
            #  定期保存模型
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
    
        # 每个epoch结束就保存一次模型
        torch.save(decoder.state_dict(), os.path.join(
            args.model_path, 'decoder-{}.ckpt'.format(epoch+1)))
        torch.save(encoder.state_dict(), os.path.join(
            args.model_path, 'encoder-{}.ckpt'.format(epoch+1)))

if __name__ == "__main__":
    # 设置参数
    parser = argparse.ArgumentParser()

    # 设置文件夹路径
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='/remote-home/competition/hwh/Code-Repository/LCRN/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='/remote-home/competition/hwh/resized2017', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='/remote-home/competition/hwh/annotations/captions_train2017.json', help='path for train annotation json file')

    # 设置打印信息步长和保存步长
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')

    # 设置模型参数
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')

    # 设置超参数
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    config = parser.parse_args()

    print(config)

    train_main(config)

import argparse
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#定义函数 Resize图像
def resize_image(image, size):
    return image.resize(size, Image.ANTIALIAS)

#定义函数Resize图像序列
def resize_images(image_dir, output_dir, size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i,image in enumerate(images):
        with open(os.path.join(image_dir,image),'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img,size)
                img.save(os.path.join(output_dir, image),img.format)
        if (i+1) % 100 == 0:
            print ("[{}/{}] Resized the images and saved into '{}".format(i+1,num_images,output_dir))

def resize_main(args):
    image_dir = args.image_dir
    output_dir = args.output_dir
    image_size = [args.image_size, args.image_size]
    resize_images(image_dir, output_dir, image_size)

# 通过argparse传参数
# 注意指定数据集相关文件的路径
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='/home/harry/Code/Image Caption/train2017/',
                        help='directory for train images')
    parser.add_argument('--output_dir', type=str, default='/home/harry/Code/Image Caption/resized2017/',
                        help='directory for saving resized images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='size for image after processing')
    config = parser.parse_args()
    resize_main(config)
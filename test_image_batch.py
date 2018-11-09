import argparse
#import time
import re
import os
import torch
from torch.utils.data import DataLoader
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


from model import Generator

class loadfromfolder(Dataset):
    def __init__(self, folder):
        super(loadfromfolder, self).__init__()
        self.folder = folder
        self.files = sorted(os.listdir(self.folder), key=lambda item: int(re.findall(r'\d+', item)[0]))
        self.filenames = [os.path.join(self.folder, x) for x in self.files]

    def __getitem__(self, index):
        return ToTensor()(Image.open(self.filenames[index]))

    def __len__(self):
        return len(self.filenames)

    def get_name(self, index):
        return self.files[index]

parser = argparse.ArgumentParser(description='Test Multiple Images')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--img_folder', type=str, help='folder with low resolution images')
parser.add_argument('--save_to', default='./', type=str, help='folder to save generated images to')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_FOLDER = opt.img_folder
SAVE_TO = opt.save_to
MODEL_NAME = opt.model_name
BATCH_SIZE = opt.batch_size

image_set = loadfromfolder(IMAGE_FOLDER)
image_loader = DataLoader(dataset=image_set, num_workers=0, batch_size=BATCH_SIZE, shuffle=False)

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load(MODEL_NAME))
else:
    model.load_state_dict(torch.load(MODEL_NAME, map_location=lambda storage, loc: storage))

img_bar = tqdm(image_loader)
count = 0
for img in img_bar:
    img_c = Variable(img, volatile=True)
    # print "SHAPE: ", img_c.shape
    if TEST_MODE:
        img_c = img_c.cuda()

    out = model(img_c)

    for im in out:
#        print "SHAPE: ", im.shape, "TYPE:", type(im)
        out_im = ToPILImage()(im.data.cpu())
        out_im.save(SAVE_TO + 'processed_' + image_set.get_name(count))
        count += 1       

    

# image = Image.open(IMAGE_NAME)
# image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
#if TEST_MODE:
#    image = image.cuda()

#start = time.clock()
#out = model(image)
#elapsed = (time.clock() - start)
#print('cost' + str(elapsed) + 's')
#out_img = ToPILImage()(out[0].data.cpu())
#out_img.save('out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)

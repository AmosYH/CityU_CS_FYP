import os
import sys
import numpy as np
import torch
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
import cv2
import torchvision.transforms as transforms

from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox

from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm

sys.path.insert(0, "/project/SCI")
from model import Finetunemodel

MODEL_FILENAME = os.environ["MODEL_FILENAME"]
MIN_FACE_WIDTH = int(os.environ["MIN_FACE_WIDTH"])
MIN_FACE_HEIGHT = int(os.environ["MIN_FACE_HEIGHT"])

save_path = '/project/static/media/results'
model_filename=f'./SCI/weights/{MODEL_FILENAME}'


class MemoryFriendlyLoaderFile(torch.utils.data.Dataset):
    def __init__(self, img_filename, task):
        self.task = task
        self.train_low_data_names = []
        self.train_low_data_names.append(img_filename)

        #self.train_low_data_names.sort()
        self.count = len(self.train_low_data_names)

        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def load_images_transform(self, file):
        im = Image.open(file).convert('RGB')
        img_norm = self.transform(im).numpy()
        img_norm = np.transpose(img_norm, (1, 2, 0))
        return img_norm

    def __getitem__(self, index):

        low = self.load_images_transform(self.train_low_data_names[index])

        h = low.shape[0]
        w = low.shape[1]

        low = np.asarray(low, dtype=np.float32)
        low = np.transpose(low[:, :, :], (2, 0, 1))

        img_name = self.train_low_data_names[index].split('/')[-1]

        return torch.from_numpy(low), img_name

    def __len__(self):
        return self.count

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True



def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def load_facedetector(network='resnet50', trained_model='./weights/Resnet50_Final.pth'):
    torch.set_grad_enabled(False)
    cfg = None
    if network == "mobile0.25":
        cfg = cfg_mnet
    elif network == "resnet50":
        cfg = cfg_re50
    # net and model
    facedetector = RetinaFace(cfg=cfg, phase = 'test')
    facedetector = load_model(facedetector, trained_model, False)
    facedetector.eval()
    print('Finished loading face detection model!')
    cudnn.benchmark = True
    device = torch.device("cuda")
    facedetector = facedetector.to(device)
    return device, facedetector, cfg

def detect_faces(source_img, device, facedetector, cfg, confidence_threshold=0.7, top_k=5000, nms_threshold=0.4, keep_top_k=750):
    img = np.float32(source_img)

    resize = 1

    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    loc, conf, landms = facedetector(img)  

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)

    return dets

def draw_dets(source_image, dets):
    result_image = source_image.copy()
    for b in dets:
        b = [0 if x < 0 else int(x) for x in b]
        if (abs(b[2] - b[0]) >= MIN_FACE_WIDTH) and (abs(b[3] - b[1]) >= MIN_FACE_HEIGHT):
            cv2.rectangle(result_image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
    return result_image


def enhance_light_image(filepath):
    os.makedirs(save_path, exist_ok=True)
    TestDataset = MemoryFriendlyLoaderFile(img_filename=filepath, task='test')

    test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1,
        pin_memory=True, num_workers=0)

    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    model = Finetunemodel(model_filename)
    model = model.cuda()
    model.eval()

    device, facedetector, cfg = load_facedetector()

    with torch.no_grad():
        for _, (input, image_name) in enumerate(test_queue):
            input = input.cuda()
            image_name = image_name[0].split('/')[-1].split('.')[0]
            i, r = model(input)
            u_name = '%s.png' % (image_name)
            print('processing {}'.format(u_name))
            u_path = save_path + '/' + u_name
            image_numpy = r[0].cpu().float().numpy()
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
            image_numpy = np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8')

            #face detection part
            dets = detect_faces(image_numpy, device, facedetector, cfg)
            result_image = draw_dets(image_numpy, dets)

            im = Image.fromarray(result_image)
            im.save(u_path, 'png')
        return save_path.split('/')[-1]+'/'+u_name

def enhance_light_video(filepath, output_filepath, output_filepath2):
    os.makedirs(save_path, exist_ok=True)

    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    model = Finetunemodel(model_filename)
    model = model.cuda()
    model.eval()

    device, facedetector, cfg = load_facedetector()

    cap = cv2.VideoCapture(
        f'uridecodebin uri=file://{filepath} ! videoconvert ! videoscale ! video/x-raw, width=1920, height=1080 ! appsink sync=0 ',
        cv2.CAP_GSTREAMER)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    if cap.isOpened():
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_filepath, fourcc, fps, (int(width)*2, int(height)))
    out2 = cv2.VideoWriter(output_filepath2, fourcc, fps, (int(width), int(height)))

    transform_list = []
    transform_list += [transforms.ToTensor()]
    transform = transforms.Compose(transform_list)

    with torch.no_grad():
        frame_num = 0
        while cap.isOpened():
            success, img_raw = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            im = Image.fromarray(img_raw)
            img_norm = transform(im).numpy()
            img_norm = np.transpose(img_norm, (1, 2, 0))
            low = np.asarray(img_norm, dtype=np.float32)
            low = np.transpose(low[:, :, :], (2, 0, 1))
            low = torch.from_numpy(low)
            low = low.cuda().unsqueeze(0)

            i, r = model(low)

            image_numpy = r[0].cpu().float().numpy()
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
            output_image = np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8')

            #face detection part
            dets = detect_faces(output_image, device, facedetector, cfg)
            result_image = draw_dets(output_image, dets)


            out.write(cv2.hconcat([img_raw, result_image]))
            out2.write(result_image)
            frame_num += 1

        out.release()

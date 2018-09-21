import sys, torch, argparse, PIL
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path
import numbers, numpy as np

lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
from xvision import transforms, draw_image_by_points
from models import obtain_model, remove_module_dict
from config_utils import load_configure
import os
import cv2

from Records.utils.terminal_utils import progressbar
from Records.utils.pointIO import *
import json
_image_path = '/home/dhruv/Projects/Datasets/Groomyfy_16k/Dlibdet/val/'
_output_path = '/home/dhruv/Projects/TFmodels/sbr/'
_pts_path_ = '/home/dhruv/Projects/Datasets/Groomyfy_16k/Dlibdet/pts/'
# _image_path = '/home/dhruv/Projects/Datasets/300VW_Dataset_2015_12_14/001/out/'



def evaluate(args):
    #amit commented to run on my system
    # assert torch.cuda.is_available(), 'CUDA is not available.'
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True


    print('The model is {:}'.format(args.model))
    snapshot = Path(args.model)
    assert snapshot.exists(), 'The model path {:} does not exist'
    # snapshot = torch.load(snapshot) #amit
    snapshot = torch.load(snapshot,map_location='cpu')

    # General Data Argumentation
    mean_fill = tuple([int(x * 255) for x in [0.485, 0.456, 0.406]])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    param = snapshot['args']
    eval_transform = transforms.Compose(
        [transforms.PreCrop(param.pre_crop_expand), transforms.TrainScale2WH((param.crop_width, param.crop_height)),
         transforms.ToTensor(), normalize])
    model_config = load_configure(param.model_config, None)
    dataset = Dataset(eval_transform, param.sigma, model_config.downsample, param.heatmap_type, param.data_indicator)
    dataset.reset(param.num_pts)

    net = obtain_model(model_config, param.num_pts + 1)
    # net = net.cuda() #amit
    weights = remove_module_dict(snapshot['state_dict'])
    nu_weights = {}
    for key, val in weights.items():
        nu_weights[key.split('detector.')[-1]] = val
        print(key.split('detector.')[-1])
    weights = nu_weights
    net.load_state_dict(weights)

    print('Prepare input data')
    images = os.listdir(args.image_path)
    total_images = len(images)

    for im_ind, aimage in enumerate(images):
        progressbar(im_ind, total_images)
        aim = os.path.join(args.image_path, aimage)
        args.image = aim
        im = cv2.imread(aim)
        imshape = im.shape
        args.face = [0, 0, imshape[0], imshape[1]]
        [image, _, _, _, _, _, cropped_size], meta = dataset.prepare_input(args.image, args.face)
        # inputs = image.unsqueeze(0).cuda() #amit
        inputs = image.unsqueeze(0)
        # network forward
        with torch.no_grad():
            batch_heatmaps, batch_locs, batch_scos = net(inputs)
        # obtain the locations on the image in the orignial size
        cpu = torch.device('cpu')
        np_batch_locs, np_batch_scos, cropped_size = batch_locs.to(cpu).numpy(), batch_scos.to(
            cpu).numpy(), cropped_size.numpy()
        locations, scores = np_batch_locs[0, :-1, :], np.expand_dims(np_batch_scos[0, :-1], -1)

        scale_h, scale_w = cropped_size[0] * 1. / inputs.size(-2), cropped_size[1] * 1. / inputs.size(-1)

        locations[:, 0], locations[:, 1] = locations[:, 0] * scale_w + cropped_size[2], locations[:, 1] * scale_h + \
                                           cropped_size[3]
        prediction = np.concatenate((locations, scores), axis=1).transpose(1, 0)

        # print ('the coordinates for {:} facial landmarks:'.format(param.num_pts))
        for i in range(param.num_pts):
            point = prediction[:, i]
            # print ('the {:02d}/{:02d}-th point : ({:.1f}, {:.1f}), score = {:.2f}'.format(i, param.num_pts, float(point[0]), float(point[1]), float(point[2])))

        if args.save:
            json_file = os.path.splitext(aimage)[0] + '.json'
            save_path = os.path.join(args.save, json_file)

            pred_pts = np.transpose(prediction, [1, 0])
            pred_pts = pred_pts[:, :-1]

            with open(save_path, 'w') as j_out:
                json.dump(pred_pts.tolist(), j_out)
            #print(pred_pts)
            #cv2.imwrite(save_path, sim)
            # image.save(args.save)
            # print ('save the visualization results into {:}'.format(args.save))
        else:
            print('ignore the visualization procedure')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a images by the trained model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, help='The snapshot to the saved detector.')
    parser.add_argument('--save', type=str, help='The path to save the visualized results.')
    parser.add_argument('--image_path', type=str, help='The path to load images from.')
    args = parser.parse_args()
    evaluate(args)

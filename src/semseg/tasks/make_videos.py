from os.path import join, isdir
from shutil import rmtree
import glob

import numpy as np

from ..data.generators import VALIDATION
from .utils import make_prediction_img, predict_x
from ..data.utils import _makedirs, save_img, zip_dir
from ..models.factory import make_model

MAKE_VIDEOS = 'make_videos'


def make_videos(run_path, options, generator):
    videos_path = join(run_path, 'videos')
    _makedirs(videos_path)

    checkpoints_path = join(run_path, 'delta_model_checkpoints')
    if not isdir(checkpoints_path):
        print('Cannot make videos without delta_model_checkpoints.')
        return

    model_paths = glob.glob(join(checkpoints_path, '*.h5'))
    model_paths.sort()
    models = []
    for model_path in model_paths:
        model = make_model(options, generator)
        model.load_weights(model_path, by_name=True)
        models.append(model)

    split_gen = generator.make_split_generator(
        VALIDATION, target_size=options.eval_target_size,
        batch_size=1, shuffle=False, augment=False, normalize=True,
        eval_mode=False)

    for video_ind, (batch_x, _) in enumerate(split_gen):
        video_path = join(videos_path, str(video_ind))
        _makedirs(video_path)
        x = np.squeeze(batch_x, axis=0)
        make_video(x, models, video_path, options, generator)

        if video_ind == options.nb_videos - 1:
            break


def make_video(x, models, video_path, options, generator):
    for frame_ind, model in enumerate(models):
        frame = make_prediction_img(
            x, options.target_size[0],
            lambda x: generator.dataset.one_hot_to_rgb_batch(
                predict_x(x, model)))

        frame_path = join(
            video_path, 'frame_{:0>4}.png'.format(frame_ind))
        save_img(frame, frame_path)

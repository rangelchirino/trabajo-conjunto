import imageio
import numpy as np
import matplotlib.pyplot as plt

# reader = imageio.get_reader('imageio:cockatoo.mp4')
# for i, im in enumerate(reader):
#     print('Mean of frame %i is %1.1f' % (i, im.mean()))

# plt.imshow(im)
# plt.show()

# fps = reader.get_meta_data()['fps']

# writer = imageio.get_writer('cockatoo_gray.mp4', fps=fps)

# for im in reader:
#     writer.append_data(im[:, :, 1])
# writer.close()

# frame1 = reader.get_data(0)
# reader.get_length()
# reader.get_meta_data()
# {'plugin': 'ffmpeg',
#  'nframes': 280,
#  'ffmpeg_version': '4.0.2 built with gcc 7.3.1 (GCC) 20180722',
#  'fps': 20.0,
#  'source_size': (1280, 720),
#  'size': (1280, 720),
#  'duration': 14.0}

def video_to_tensor(video):
    reader = imageio.get_reader(video)
    frames = []
    for fr in reader:
        frames.append(fr)
    frames = np.asarray(frames)
    reader.close()
    return frames

video = 'imageio:cockatoo.mp4'
frames = video_to_tensor(video)
frames.shape
# (280, 720, 1280, 3) = (n_frames, height, width, channels)

plt.imshow(frames[0])
plt.show()

# import tensorly as tl
# from tensorly import decomposition

# tensor = tl.tensor(frames)
# tensor = frames.astype('float64')

# cp_factors = decomposition.parafac(frames, rank=20)

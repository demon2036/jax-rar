import glob
import os
import threading

import PIL
import jax
import numpy as np
import webdataset as wds
from PIL import Image
from webdataset import TarWriter



def collect_process_data(data):
    local_data = []
    local_devices = jax.local_devices()

    for shard in data.addressable_shards:
        device = shard.device
        local_shard = shard.data
        if device in local_devices:
            # if jax.process_index() == 0:
            #     print(device, local_devices)
            local_data.append(np.array(local_shard))
    local_data = np.concatenate(local_data, axis=0)
    return local_data


counter=0

def thread_write(images, class_labels, sink):
    # images = np.array(images)
    # class_labels = np.array(class_labels)

    images=collect_process_data(images)
    class_labels = collect_process_data(class_labels)
    global counter

    for img, cls_label in zip(images, class_labels):
        sink.write({
            "__key__": "%010d" % counter,
            "jpg": PIL.Image.fromarray(img),
            "cls": int(cls_label),
        })
        counter += 1
    if jax.process_index() == 0:
        print(counter, images.shape)



def thread_write_vq(tokens, class_labels,siglip_features,dino_features, sink):
    # images = np.array(images)
    # class_labels = np.array(class_labels)

    tokens=np.array(tokens,dtype=np.int16)
    class_labels = np.array(class_labels)
    siglip_features=np.array(siglip_features,dtype=np.float32)
    dino_features=np.array(dino_features,dtype=np.float32)

    global counter

    for token, cls_label,siglip_feature,dino_feature in zip(tokens, class_labels,siglip_features,dino_features):
        # print(siglip_feature.shape,dino_feature.shape,token.dtype,cls_label.dtype,siglip_features.dtype,dino_features.dtype)
        # print(cls_label)
        # print(type(token),type(cls_label),type(siglip_features),type(dino_feature))

        sink.write({
            "__key__": "%010d" % counter,
            "token.pyd": token,
            "cls": int(cls_label),
            "siglip_feature.pyd": siglip_feature,
            'dino_feature.pyd':dino_feature
        })
        counter += 1
    # if jax.process_index() == 0:
    #     print(counter, images.shape)

class CustomShardWriter(wds.ShardWriter):
    """
    CustomShardWriter to make it suitable for increase shard counter step by  jax.process_count()
    """
    def __init__(self, progress_count, *args, **kwargs):
        self.progress_count = progress_count
        super().__init__(*args, **kwargs)

    def next_stream(self):
        # print('her is next stream')
        """Close the current stream and move to the next."""
        self.finish()
        self.fname = self.pattern % self.shard
        if self.verbose:
            print(
                "# writing",
                self.fname,
                self.count,
                "%.1f GB" % (self.size / 1e9),
                self.total,
            )
        self.shard += self.progress_count
        self.tarstream = TarWriter(self.fname, **self.kw)
        self.count = 0
        self.size = 0


def get_remote_path(remote_path):
    dst = remote_path
    if 'gs' not in remote_path:
        dst = os.getcwd() + '/' + dst
        os.makedirs(dst, exist_ok=True)
    return dst

def send_file(keep_files, dst, rng, iter):

    files = glob.glob('shard_path/*.tar')
    files.sort(key=lambda x: os.path.getctime(x), )
    send_file_threads=[]
    ckpt={
        'rng': rng,
        'label': iter - keep_files
    }

    if len(files) == 0:
        raise NotImplemented()
    elif len(files) <= keep_files:
        pass
    else:
        if keep_files == 0:
            files = files
        else:
            files = files[:-keep_files]


        for file in files:
            base_name = os.path.basename(file)
            if jax.process_index() == 0:
                print(base_name, files)

            def send_data_thread(src_file, dst_file):
                with wds.gopen(src_file, "rb") as fp_local:
                    data_to_write = fp_local.read()

                with wds.gopen(f'{dst_file}', "wb") as fp:
                    fp.write(data_to_write)
                    # fp.flush()

                os.remove(src_file)

            thread=threading.Thread(target=send_data_thread, args=(file, f'{dst}/{base_name}'))
            thread.start()
            send_file_threads.append(thread)
            send_file_threads.append(thread)


    return ckpt,send_file_threads


        #     # orbax_checkpointer = ocp.PyTreeCheckpointer()
        #     save_args = orbax_utils.save_args_from_target(ckpt)
        #     checkpointer.save(f'{dst}/resume.json', ckpt, save_args=save_args, force=True)

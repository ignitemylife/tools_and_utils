# -*- coding:utf-8

"""
demo for multi process
"""
import os
import sys
from tqdm import tqdm

import math
import time
import multiprocessing
import argparse


def get_value(v, t):
    if t in ["i", "integer"]:
        return int(v)
    else:
        return float(v)


class MultiProcess:
    """MultiProcess, Give total samples to deal with , a function and kwargs and split parts

    :parameter
    ----------
    @samples: Total sampels to process , in list or tuple type.
    @function: A single thread func to process samples:
    @num_process: number of process to run
    @feed_back: `func` return values, it can be list or tuple, only inter and double supported.
        eg['i','d'] which mean an inte and double while be return from func
    @kwargs: Other parameters passing to func

    :method
    ----------
    @start: Star processing
    @wait: wait for all the process to exit
    @run:  combine `start` and `wait`

    :Propreties
    ----------
    @feed_back: `function` feed backs, in original order. shape[num_process,len(`feed_back`')

    :How2Use
    ----------
    samples = datas
    function = task_funC
    # if `func` doesnot return values
    MultiProcess(samples,func,num_process=8).run()
    # Else
    mp = MultiProcess(samples,func,num_process=8,feed_back=['i','d'])
    mp.run()
    feed_back = mp.feed_back()
    """

    def __init__(self, samples, function, num_process=8, feed_back=[], **kwargs):
        self.samples = samples
        self.func = function
        self.kwargs = kwargs
        self.num_process = min(num_process, len(samples))
        self.thread_list = []

        self.share_mem = []
        for t in feed_back:
            if not t in ["i", "integer", "double", "d"]:
                raise ValueError("Type must be i(integer) of d(double).")

        for _ in range(self.num_process):
            fbv = []
            for _ in range(len(feed_back)):
                fbv.append(multiprocessing.Value("d", 0))
            self.share_mem.append(fbv)
        self.feed_back_type = feed_back

    def start(self, sleep_time=0):
        """
        :exception
        ----------
        ValueError: `sampels`,`func`,`num_process` parameters setting
        """
        if self.samples and self.func and self.num_process:
            samples_per_thread = int(math.ceil(len(self.samples) / self.num_process))
            for i in range(self.num_process):
                kwargs = {
                    "thread_id": i,
                    "samples": self.samples[
                        i * samples_per_thread : i * samples_per_thread
                        + samples_per_thread
                    ],
                }
                if self.feed_back_type:
                    kwargs.update({"feed_back": self.share_mem[i]})
                kwargs.update(self.kwargs)
                sthread = multiprocessing.Process(target=self.func, kwargs=kwargs)
                sthread.start()
                self.thread_list.append(sthread)
                if sleep_time and sleep_time > 0:
                    time.sleep(sleep_time)
        else:
            raise ValueError(
                "Erorr while start multi process please check samples , func, num_process parameters"
            )

    def wait(self):
        for t in self.thread_list:
            t.join()

    def run(self, sleep_time=0):
        self.start(sleep_time)
        self.wait()
        return self

    @property
    def feed_back(self):
        fb_list = []
        for fb in self.share_mem:
            fb_list += [get_value(v, t) for v, t in zip(fb, self.feed_back_type)]
        return fb_list


def _task_dw(samples: list, thread_id: int):
    n = len(samples)
    for i in tqdm(range(n), desc="process %s" % thread_id):
        dw_url = samples[i].rstrip()
        try:
            # os.system("youtube-dl -x --audio-format wav %s" % dw_url)
            os.system("yt-dlp -x %s" % dw_url)
        except Exception as e:
            continue


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename')
    parser.add_argument('--urls', nargs='+')
    parser.add_argument('--dst')
    parser.add_argument('--nproc', type=int, default=8)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    filename = args.filename
    urls = args.urls
    if filename is not None:
        dw_video_list = open(filename).readlines()
    elif urls is not None:
        dw_video_list = urls
    else:
        raise ValueError('please check inpurs')

    to_download = dw_video_list

    num_process = args.nproc\

    print("total image count = %d" % len(to_download))
    MultiProcess(
        samples=to_download, function=_task_dw, num_process=num_process
    ).run()
